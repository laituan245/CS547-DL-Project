import torch
import torch.nn as nn
from dataset import *

class BNConv1D(nn.Module):

    def __init__(self, input_dim=300, output_dim=16, kernel_size=3, padding=1, stride=1):
        super(BNConv1D, self).__init__()
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, stride, padding)

    def forward(self, input):
        return self.conv(self.batchnorm(self.relu(input)))

class BottlenetConv(nn.Module):
    def __init__(self, growth_rate=16, input_dim=300, kernel_size=3, padding=1, stride=1):
        super(BottlenetConv, self).__init__()
        self.conv0 = BNConv1D(input_dim, growth_rate*4, 1, 0, 1)
        self.conv1 = BNConv1D(growth_rate*4, growth_rate, kernel_size, padding, stride)
    def forward(self, input):
        output = self.conv0(input)
        output = self.conv1(output)
        return torch.cat([input, output], 1)

class DenseBlock(nn.Module):
    def __init__(self, growth_rate=16, layer=5, input_dim=300, kernel_size=3, padding=1, stride=1):
        super(DenseBlock, self).__init__()
        self.fc = nn.Sequential()
        dim=input_dim
        for _ in range(layer):
            self.fc.add_module(BottlenetConv(growth_rate, dim, kernel_size, padding, stride))
            dim=dim+growth_rate
    def forward(self, input):
        return self.fc(input)

class DenseNet(nn.Module):
    def __init__(self, growth_rate=16, layer=5, input_dim=300, kernel_size=3, padding=1, stride=1):
        pass
    def forward(self, input):
        pass
