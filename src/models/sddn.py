import torch 
import torch.nn as nn
import torch.nn.functional as F
from operator import mul
import functools
import math
from config import Config

class GeneratorModule(nn.Module):
    """
    This is a subclass of Pytorch's nn.Module with a generate method. 
    This is needed for SDDNs where the generate pass is different from either the 
    train or forwards pass.
    """
    def __init__(self):
        super().__init__()
    def generate(self, *args, **kwargs):
        raise NotImplementedError("Generate method must be overridden")

class SddnSelect(GeneratorModule):
    """
    This layer performs the selection step of Splitable Discrete Distribution Networks.

    Parmeters:
    k is the number of outputs that the discrete operation will pick from.
    loss_function is a torch module that may have learnable parameters. It 
    must take in inputs of shape broadcastable to (batch_size * k, output dimension, ...) and have an output of shape 
    (batch_size * k). The reduction over the other dimensions must be a mean reduction. The loss must be a minimizer of NLL,
    such as MSE for real-valued variables or cross-entropy.

    Inputs: 
    The first input, x, is a tensor of shape (Batch size, output dimension, ...). 
    The second input, target is of shape (Batch size, output dimension, ...)

    The output is a tuple of 2 tensors. They have shape
    [(Batch size, output dimension, ...), (Batch size), (Batch size)]. 
        The first is the selected output. 
        The second is the loss associated with prediction error to the chosen outputs. 
        This loss is a probability weighted average for lower variance. 
        The third is the KL divergence between the predicted output distribution and the actual output distribution.

    Generate method:
        Takes in input, generates output of the same shape.
    
    """
    def __init__(self, cfg: Config, loss_function):
        super().__init__()
        self.cfg = cfg
        self.k = cfg.k
        self.loss_function = loss_function
        #This maintains a running average of the pick_frequency each entry is chosen at.
        self.pick_frequency = nn.Parameter(torch.full((cfg.k,), 7.0), requires_grad=False)
        self.centers = nn.Conv1d(cfg.inner_dim, cfg.inout_dim * cfg.k, 1)
        self.selection_estimator = nn.Conv1d(cfg.inner_dim, cfg.k, 1)
        self.pick_exp_factor = .97
        self.selection_target_weight = 5
        self.rebalance = True
        self.print_count = 0

    """
    Takes in x and target. Returns a tensor of shape (Batch Size, k)
    """
    def compute_per_entry_loss(self, x_sizes, x, target):
        #Reshape to put all k outputs of a given item into the batch dimension.
        batched_x = x.reshape((x_sizes[0] * self.k, self.cfg.inout_dim, *x_sizes[2:]))
        target = target.repeat_interleave(self.k, dim = 0)
        loss = self.loss_function.forward(batched_x, target) 
        loss = loss.reshape(x_sizes[0], self.k)
        return loss

    def sample_one_hot(self, selection_weights):
        selection_index = torch.multinomial(selection_weights, 1).squeeze(dim=1)
        selection_mask = F.one_hot(selection_index, num_classes = self.k)
        if self.training:
            self.pick_metrics(selection_mask)
        return selection_mask

    def pick_metrics(self, mask):
        selected_count = torch.sum(mask, dim = 0)
        self.pick_frequency *= self.pick_exp_factor
        self.pick_frequency += (1-self.pick_exp_factor) * selected_count
        if self.print_count > 20:
            torch.set_printoptions(sci_mode=False)
            print(self.pick_frequency)
            torch.set_printoptions(sci_mode=True)
            self.print_count = 0 
        else:
            self.print_count += 1
    """
    Takes in x and target. Returns a tensor of shape (Batch Size, k)
    """
    def apply_selections(self, sizes, x, selections):
        x = x.reshape((sizes[0], self.k, self.cfg.inout_dim, *sizes[2:]))
        selections = selections.reshape((sizes[0], self.k, *[1 for _ in range(2, x.dim())]))
        selected = torch.sum(x * selections, dim = 1)
        return selected
    
    def conv_layers(self, x):
        sizes = x.size() #x is (Batch Size, in_features, image_dimensions)
        x = x.reshape(sizes[0], sizes[1], -1)
        centers = self.centers(x)
        selection_logits = torch.mean(self.selection_estimator(x), dim=2)
        log_selection_estimate = F.log_softmax(selection_logits, dim=1)   
        return centers, log_selection_estimate
    
    def forward(self, x, target):
        sizes = x.size() #x is (Batch Size, in_features, ...)
        #x will be (Batch Size, out_features * k, sequence_dimension)
        #log_selection_estimate is now (Batch Size, k)
        (centers, log_selection_estimate) = self.conv_layers(x) 
        per_entry_loss = self.compute_per_entry_loss(sizes, centers, target) #(Batch Size, k)
        log_seletion_target = F.log_softmax(per_entry_loss * -self.selection_target_weight, dim=1) #(Batch Size, k)
        selection_weights = torch.exp(log_seletion_target) #(Batch Size, k)
        selection_kl_div = torch.sum(selection_weights * (log_seletion_target - log_selection_estimate), dim = 1) #(Batch Size)
        selections = self.sample_one_hot(selection_weights) #(Batch Size, k) - One hots
        output = self.apply_selections(sizes, centers, selections)
        selected_loss = torch.sum(per_entry_loss * selections, dim=1) #(Batch Size)
        #Since we want average loss per pixel or dim, and the information from the mean is global, 
        #it needs to be dividied by the number of pixels. Since there is a KL divergence, 
        #that takes into account the effects of k
        loss = selected_loss + selection_kl_div/functools.reduce(mul, centers[2], 1)
        return (output, selected_loss, selection_kl_div)

    def generate(self, x):
        (centers, log_selection_estimate) = self.conv_layers(x) 
        selections = self.sample_one_hot(torch.exp(log_selection_estimate)) #(Batch Size, k) - One hots
        out = self.apply_selections(x.size(), centers, selections)
        return out
    
    """
    Takes in the loss tensor. Returns a 1 hot tensor for the minimum loss index. 
    This also updates running averages of pick frequency if in train mode.
    Returns a tensor of shape (Batch Size, k)
    """
    def process_min_loss_mask(self, loss):
        #Since we are taking the min, decrease loss for less frequently picked items to help balance classes
        if self.rebalance:
            (_, min_loss_index) = torch.min(loss * self.pick_frequency.reshape((1, self.k)), dim=1)
        else:
            (_, min_loss_index) = torch.min(loss, dim=1)
        min_loss_mask = F.one_hot(min_loss_index, num_classes = self.k)

        return min_loss_mask

class SddnMseLoss(GeneratorModule):
    """
    This layer is an MSE loss for use in SDDN.

    Parmeters:
    weight: a multiplier for the loss.

    Inputs/Outputs:
    Any tensors that are broadcastable to each other and have dim>=2

    """
    def __init__(self, weight) -> None:
        super().__init__()
        
        self.weight = weight

    def forward(self, x, target):
        diff = (x - target)
        return self.weight * torch.mean(diff * diff, list(range(1,diff.dim())))
    

class SddnCrossEntropyLoss(GeneratorModule):
    """
    This layer is an Log Likelihood loss for use in SDDN.

    Parmeters:
    weight: a multiplier for the loss.

    Inputs/Outputs:
    Any tensors that are broadcastable to each other and have dim>=2

    """
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, target):
        loss = self.loss(logits, target)
        return torch.mean(loss, list(range(1, loss.dim())))
    

class SddnCrossEntropySelect(GeneratorModule):
    """
    This layer is an SDDN block with Cross Entropy loss.

    Parmeters:
    k is the number of outputs that the discrete operation will pick from.

    Inputs/Outputs:
    Same as SDDNSelect

    """
    def __init__(self, in_features, out_features, k) -> None:
        super().__init__()
        self.loss = SddnCrossEntropyLoss() #Loss scaling based on noise scale
        self.select = SddnSelect(in_features, out_features, k, self.loss)

    def forward(self, x, target):
        return self.select.forward(x, target)

    def generate(self, x):
        return self.select.generate(x)

class SddnMseSelect(GeneratorModule):
    """
    This layer is an SDDN block with MSE loss.

    Parmeters:
    k is the number of outputs that the discrete operation will pick from.

    Inputs/Outputs:
    Same as SDDNSelect

    """
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.loss = SddnMseLoss(1/(2 * cfg.noise**2)) #Loss scaling based on noise scale
        self.select = SddnSelect(cfg, self.loss)

    def forward(self, x, target):
        return self.select.forward(x, target)

    def generate(self, x):
        return self.select.generate(x)

class SddnMseBlockFC(GeneratorModule):
    """
    An SDDN block with fully connected layers
    """
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.project = nn.Linear(cfg.inout_dim, cfg.inner_dim)
        self.big_lin = nn.Linear(cfg.inner_dim, cfg.inner_dim)
        self.sddn = SddnMseSelect(cfg)

    def calculate_x(self, x):
        x = F.relu(self.project(x))
        x = F.relu(self.big_lin(x))
        return x

    def forward(self, x, target):
        x = self.calculate_x(x)
        return self.sddn(x, target)

    def generate(self, x):
        x = self.calculate_x(x)
        return self.sddn.generate(x)

class SddnFc(GeneratorModule):
    """
    Defines a fully connected SDDN network. 

    Params
    num_blocks: The number of SDDN blocks
    inout_dim: The dimensionality of the input/output
    inner_dim: The dimensionality of the inner linear layer in each block
    k: The number of possible splits.

    Inputs: input data and target output. 
    Outputs: Prediction and per-layer losses. 
    """
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.base = nn.Parameter(torch.zeros((1, cfg.inout_dim)))
        self.blocks = nn.ModuleList([SddnMseBlockFC(cfg) for _ in range(cfg.num_blocks)])

    def forward(self, target):
        x = self.base.expand((target.size()[0], -1))
        losses=[]
        for block in self.blocks:
            out = block(x, target)
            x = out[0]
            losses.append(out[1])
        return x, losses

    def generate(self, batch_size):
        with torch.no_grad():
            x = self.base.expand((batch_size, -1))
            for block in self.blocks:
                x = block.generate(x)
            x = x + torch.randn_like(x) * self.cfg.noise
            return x