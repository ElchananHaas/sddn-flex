import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.v2 as transforms
from sddn import SddnCrossEntropySelect, GeneratorModule
import matplotlib.pyplot as plt
import argparse 
import math

class SddnCEBlockConv(GeneratorModule):
    """
    An SDDN block with convolutional layers. 
    """
    def __init__(self, inout_dim, inner_dim, k) -> None:
        super().__init__()
        self.project = nn.Conv2d(inout_dim, inner_dim, 3, padding = 'same')
        self.big_lin = nn.Conv2d(inner_dim, inner_dim, 3, padding = 'same')
        self.to_out = nn.Conv2d(inner_dim, inout_dim * k, 1, padding = 'same')
        self.sddn = SddnCrossEntropySelect(k)

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

class SddnConv(GeneratorModule):
    """
    Defines a convolutional SDDN network, suitable for MNIST. 

    Params
    num_blocks: The number of SDDN blocks
    inout_dim: The dimensionality of the input/output
    inner_dim: The dimensionality of the inner linear layer in each block
    k: The number of possible splits.

    Inputs: input data and target output. 
    Outputs: Prediction and per-layer losses. 
    """
    def __init__(self, num_blocks, inout_dim, w, h, inner_dim, k) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.zeros((1, inout_dim, w, h)))
        self.blocks = nn.ModuleList([SddnCEBlockConv(inout_dim, inner_dim, k) for _ in range(num_blocks)])

    def forward(self, target):
        x = self.base.expand((target.size()[0], -1, -1, -1))
        losses=[]
        for block in self.blocks:
            out = block(x, target)
            x = out[0]
            losses.append(out[1])
        return x, losses

    def generate(self, batch_size):
        with torch.no_grad():
            x = self.base.expand((batch_size, -1, -1, -1))
            for block in self.blocks:
                x = block.generate(x)
            x = F.softmax(x, dim = 1)
            #We need to put channel last for sampling
            x = x.permute(0, 2, 3, 1)
            size = x.size()
            x = x.reshape(-1, size[3])
            x = torch.multinomial(x, 1)
            x = x.reshape(size[0:3])
            return x

parser = argparse.ArgumentParser(
                    prog='SDDN SKlearn datasets',
                    description='Test Splitable Discrete Distribution Networks on MNIST or Fashion MNIST datasets')
parser.add_argument('--dataset', choices = ['mnist'], default = 'mnist')
parser.add_argument('-k', type = int, default = 10)
parser.add_argument('--num-blocks', type = int, default = 1)
args = parser.parse_args()

transform = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.long)])

train_dataset = torchvision.datasets.MNIST(
    root='./external_data', 
    train=True,
    download=True,
    transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./external_data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
model = SddnConv(num_blocks = args.num_blocks, inout_dim = 256, w = 28, h = 28, inner_dim = 256, k = args.k)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

REPORT_INTERVAL = 1
DISPLAY_INTERVAL = 10

def plot(model, n_samples):
    next_square = math.ceil(n_samples**.5)
    generated = model.generate(n_samples)
    fig, ax = plt.subplots(next_square, next_square)
    for i in range(generated.size()[0]):
        data = generated[i]
        ax[i//next_square][i % next_square].imshow(data)
    plt.show()
    plt.close(fig)

for count, (batch, _labels) in enumerate(train_loader):
    batch = torch.squeeze(batch, dim = 1)
    optimizer.zero_grad()
    (prediction, losses) = model(batch)
    loss = torch.mean(torch.stack(losses, dim=-1))
    loss.backward()
    optimizer.step()
    if count % REPORT_INTERVAL == REPORT_INTERVAL - 1:
        print(loss.item())
    if count % DISPLAY_INTERVAL == DISPLAY_INTERVAL - 1:
        plot(model, 9)