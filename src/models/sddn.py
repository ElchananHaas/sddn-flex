import torch 
import torch.nn as nn
import torch.nn.functional as F
from operator import mul
import functools
import math

class SddnSelect(nn.Module):
    """
    This layer performs the selection step of Splitable Discrete Distribution Networks.

    Parmeters:
    k is the number of outputs that the discrete operation will pick from.
    loss_function is a torch module that may have learnable parameters. It 
    must take in inputs of shape broadcastable to (batch_size, k, output dimension, ...) and have an output of shape 
    (batch_size, k). The reduction over the other dimensions must be a mean reduction. The loss must be a minimizer of NLL,
    such as MSE for real-valued variables or cross-entropy.

    Inputs: 
    The first input, x, is a tensor of shape (Batch size, output dimension * k, ...). 
    The second input, target is of shape (Batch size, output dimension, ...)

    The output is a tuple of 2 tensors. They have shape
    [(Batch size, output dimension, ...), (Batch size)]. The first is the selected output. The second is the loss.
    The loss has an additional log_2(K)/(output dimensionality) term added to account for the information provided by the choice operation.

    """
    def __init__(self, k, loss_function) -> None:
        super().__init__()
        self.k = k
        self.loss_function = loss_function

    def forward(self, x, target):
        sizes = x.size()
        target_shape = (sizes[0], self.k, sizes[1] // self.k, *sizes[2:])
        x = x.reshape(target_shape)
        target = target.unsqueeze(dim = 1)
        elements_per = functools.reduce(mul, sizes[2:], 1)
        loss = self.loss_function.forward(x ,target) + math.log(self.k, 2)/elements_per
        (min_loss, min_loss_index) = torch.min(loss, dim=1)
        min_loss_mask = F.one_hot(min_loss_index, num_classes = self.k)
        min_loss_mask = min_loss_mask.reshape((sizes[0], self.k, *[1 for _ in range(2, x.dim())]))
        selected = torch.sum(x * min_loss_mask, dim = 1)
        return (selected, min_loss)

class SddnMseLoss(nn.Module):
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
        return self.weight * torch.mean(diff * diff, list(range(2, diff.dim())))
    
class SddnMseSelect(nn.Module):
    """
    This layer is an SDDN block with MSE loss.

    Parmeters:
    k is the number of outputs that the discrete operation will pick from.

    Inputs/Outputs:
    Same as SDDNSelect

    """
    def __init__(self, k) -> None:
        super().__init__()
        self.loss = SddnMseLoss(12.5) #Multipler of 12.5 for standard deviation of 0.2
        self.select = SddnSelect(k, self.loss)

    def forward(self, x, target):
        return self.select.forward(x, target)

class SddnMseBlockFC(nn.Module):
    """
    An SDDN block with fully connected layers
    """
    def __init__(self, inout_dim, inner_dim, k) -> None:
        super().__init__()
        self.project = nn.Linear(inout_dim, inner_dim)
        self.big_lin = nn.Linear(inner_dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, inout_dim * k)
        self.blocks = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
        self.sddn = SddnMseSelect(k)
    def forward(self, x, target):
        x = F.relu(self.project(x))
        x = F.relu(self.big_lin(x))
        x = self.to_out(x)
        return self.sddn(x, target)

class SddnFc(nn.Module):
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
    def __init__(self, num_blocks, inout_dim, inner_dim, k) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.zeros((1, inout_dim)))
        self.blocks = nn.ModuleList([SddnMseBlockFC(inout_dim, inner_dim, k) for _ in range(num_blocks)])

    def forward(self, target):
        x = self.base.expand((target.size()[0], -1))
        losses=[]
        for block in self.blocks:
            out = block(x, target)
            x = out[0]
            losses.append(out[1])
        return x, losses