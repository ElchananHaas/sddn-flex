import torch
from sklearn import datasets
from sddn import SddnFc
import argparse 
import matplotlib.pyplot as plt
from config import Config

parser = argparse.ArgumentParser(
                    prog='SDDN SKlearn datasets',
                    description='Test Splitable Discrete Distribution Networks on SKLearn toy datasets')
parser.add_argument('--dataset', choices = ['moons'], default = 'moons')
parser.add_argument('-k', type = int, default = 10)
parser.add_argument('--inner-dim', type = int, default = 20)
parser.add_argument('--num-blocks', type = int, default = 1)
parser.add_argument('--noise', type = float, default = 0.2)
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--inout-dim', type = int, default = 2)
parser.add_argument('--split-threshold', type = float, default = .01)
cfg = Config()
parser.parse_args(namespace=cfg)

model = SddnFc(cfg)
optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)

def plot(model, n_samples):
    data, _ = datasets.make_moons(n_samples=n_samples, noise=cfg.noise)
    generated = model.generate(n_samples)
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], marker='o', color = 'orange') 
    ax.scatter(generated[:, 0], generated[:, 1], marker='o', color = 'blue') 
    plt.show()
    plt.close(fig)

REPORT_INTERVAL = 10
DISPLAY_INTERVAL = 1000
for i in range(1000000):
    data, _ = datasets.make_moons(n_samples=200, noise=cfg.noise)
    data = torch.tensor(data)
    optimizer.zero_grad()
    (prediction, losses) = model(data)
    loss = torch.mean(torch.stack(losses, dim=-1))
    loss.backward()
    optimizer.step()
    if i % REPORT_INTERVAL == REPORT_INTERVAL - 1:
        print(loss.item())
    if i % DISPLAY_INTERVAL == DISPLAY_INTERVAL - 1:
        plot(model, 100)

