import torch 
import torch.nn as nn
import torch.nn.functional as F
from operator import mul
import functools
import math

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
    The first input, x, is a tensor of shape (Batch size, output dimension * k, ...). 
    The second input, target is of shape (Batch size, output dimension, ...)

    The output is a tuple of 2 tensors. They have shape
    [(Batch size, output dimension, ...), (Batch size)]. The first is the selected output. The second is the loss.
    The loss has an additional log_2(K)/(output dimensionality) term added to account for the information provided by the choice operation.

    Generate method:
    This method selects one of the k outputs uniformly at random.

    """
    def __init__(self, k, loss_function):
        super().__init__()
        self.k = k
        self.loss_function = loss_function
        #This maintains a running average of the pick_frequency each entry is chosen at.
        self.pick_frequency = nn.Parameter(torch.full((k,), 7), requires_grad=False)
        self.pick_exp_factor = .98
        self.rebalance = True
    """
    Takes in x and target. Returns a tensor of shape (Batch Size, k)
    """
    def compute_loss(self, x_sizes, x, target):
        #Reshape to put all k outputs of a given item into the batch dimension.
        batched_x = x.reshape((x_sizes[0] * self.k, x_sizes[1] // self.k, *x_sizes[2:]))
        target = target.repeat_interleave(self.k, dim = 0)
        #This addition to the loss accounts for information provided through the min operation.
        provided_penalty = math.log(self.k, 2)/functools.reduce(mul, x_sizes[2:], 1)
        loss = self.loss_function.forward(batched_x, target) + provided_penalty
        loss = loss.reshape(x_sizes[0], self.k)
        return loss

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
        if self.training:
            selected_count = torch.sum(min_loss_mask, dim = 0)
            self.pick_frequency = nn.Parameter(self.pick_frequency * self.pick_exp_factor + selected_count * (1 - self.pick_exp_factor), requires_grad=False)
            print(self.pick_frequency)
        return min_loss_mask

    def forward(self, x, target):
        sizes = x.size()
        loss = self.compute_loss(sizes, x, target)
        min_loss_mask = self.process_min_loss_mask(loss)
        #Use the losses of the items actually picked, not just the min of the loss.
        #This is needed if frequencies are being balanced externally.
        min_loss = torch.sum(loss * min_loss_mask, dim = 1) 
        #The broadcasting here is lining up k on index 1, then summing over it. 
        #This selects the masked value.
        x = x.reshape((sizes[0], self.k, sizes[1]//self.k, *sizes[2:]))
        min_loss_mask = min_loss_mask.reshape((sizes[0], self.k, *[1 for _ in range(2, x.dim())]))
        selected = torch.sum(x * min_loss_mask, dim = 1)
        return (selected, min_loss)

    def generate(self, x):
        sizes = x.size()
        target_shape = (sizes[0], self.k, sizes[1] // self.k, *sizes[2:])
        x = x.reshape(target_shape)
        selection_index = torch.randint(self.k, (sizes[0],))
        selection_mask = F.one_hot(selection_index, num_classes = self.k)
        selection_mask = selection_mask.reshape((sizes[0], self.k, *[1 for _ in range(2, x.dim())]))
        selected = torch.sum(x * selection_mask, dim = 1)
        return selected

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
    def __init__(self, k) -> None:
        super().__init__()
        self.loss = SddnCrossEntropyLoss() #Loss scaling based on noise scale
        self.select = SddnSelect(k, self.loss)

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
    def __init__(self, k, noise) -> None:
        super().__init__()
        self.loss = SddnMseLoss(1/(2 * noise**2)) #Loss scaling based on noise scale
        self.select = SddnSelect(k, self.loss)

    def forward(self, x, target):
        return self.select.forward(x, target)

    def generate(self, x):
        return self.select.generate(x)

class SddnMseBlockFC(GeneratorModule):
    """
    An SDDN block with fully connected layers
    """
    def __init__(self, inout_dim, inner_dim, k, noise) -> None:
        super().__init__()
        self.project = nn.Linear(inout_dim, inner_dim)
        self.big_lin = nn.Linear(inner_dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, inout_dim * k)
        self.sddn = SddnMseSelect(k, noise)

    def calculate_x(self, x):
        x = F.relu(self.project(x))
        x = F.relu(self.big_lin(x))
        x = self.to_out(x)
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
    def __init__(self, num_blocks, inout_dim, inner_dim, k, noise) -> None:
        super().__init__()
        self.noise = noise
        self.base = nn.Parameter(torch.zeros((1, inout_dim)))
        self.blocks = nn.ModuleList([SddnMseBlockFC(inout_dim, inner_dim, k, noise) for _ in range(num_blocks)])

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
            x = x + torch.randn_like(x) * self.noise
            return x