import torch
from sklearn import datasets
from sddn import SddnFc, SddnSelect, SddnMseLoss
import argparse 
import matplotlib.pyplot as plt
from config import Config

parser = argparse.ArgumentParser(
                    prog='SDDN SKlearn datasets',
                    description='Test Splitable Discrete Distribution Networks on SKLearn toy datasets')
parser.add_argument('--dataset', choices = ['moons'], default = 'moons')
parser.add_argument('-k', type = int, default = 3)
parser.add_argument('--inner-dim', type = int, default = 5)
parser.add_argument('--num-blocks', type = int, default = 1)
parser.add_argument('--noise', type = float, default = 0.2)
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--inout-dim', type = int, default = 2)
parser.add_argument('--split-threshold', type = float, default = .01)
cfg = Config()
parser.parse_args(namespace=cfg)



loss = SddnMseLoss(1)
layer = SddnSelect(cfg, loss)

input = torch.randn((1, 5, 1))
target = torch.randn((1, 2, 1))
print(layer.conv_layers(input))
layer.split(0, 1)
print(layer.conv_layers(input))