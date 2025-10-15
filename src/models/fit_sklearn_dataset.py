# /// script
# requires-python = "==3.13"
# dependencies = [
#     "torch",
#     "matplotlib",
#     "scikit-learn"
# ]
# ///

import torch
from sklearn import datasets
from sddn import SddnFc
import argparse 

parser = argparse.ArgumentParser(
                    prog='SDDN SKlearn datasets',
                    description='Test Splitable Discrete Distribution Networks on SKLearn toy datasets')
parser.add_argument('--dataset', choices = ['moons'], default = 'moons')
parser.add_argument('-k', type = int, default = 10)
parser.add_argument('--inner-dim', type = int, default = 20)
parser.add_argument('--num-blocks', type = int, default = 1)
args = parser.parse_args()

model = SddnFc(num_blocks = args.num_blocks, inout_dim = 2, inner_dim = args.inner_dim, k = args.k)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

REPORT_INTERVAL = 10
for i in range(1000000):
    data, _ = datasets.make_moons(n_samples=200, noise=0.2)
    data = torch.tensor(data)
    optimizer.zero_grad()
    (prediction, losses) = model(data)
    loss = torch.mean(torch.stack(losses, dim=-1))
    loss.backward()
    optimizer.step()
    if i % REPORT_INTERVAL == REPORT_INTERVAL - 1:
        print(loss.item())
print(X.shape)

