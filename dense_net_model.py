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
        for i in range(layer):
            self.fc.add_module("conv"+str(i), BottlenetConv(growth_rate, dim, kernel_size, padding, stride))
            dim=dim+growth_rate
    def forward(self, input):
        return self.fc(input)

class FullyConnect(nn.Module):
    def __init__(self, input_dim, output_dim, internal_dim=1000):
        super(FullyConnect, self).__init__()
        self.relu = nn.ReLU()
        self.l0 = nn.Linear(input_dim, internal_dim)
        self.l1 = nn.Linear(internal_dim, internal_dim)
        self.l2 = nn.Linear(internal_dim, output_dim)
    def forward(self, input):
        input = input.reshape(-1, 652)
        output = self.relu(self.l0(input))
        output = self.relu(self.l1(output))
        output = self.l2(output)
        return output

class DenseNet(nn.Module):
    def __init__(self, input_length=1024, conv_dim=12, growth_rate=32, layer=5, output_class=2):
        super(DenseNet, self).__init__()
        self.conv0 = BNConv1D(1, conv_dim, 7, 3, 2)
        self.maxpool = nn.MaxPool1d(3, 2, 1)

        self.db0 = DenseBlock(growth_rate, layer, conv_dim, 3, 1, 1)
        self.tranconv0 = BNConv1D(growth_rate*layer+conv_dim, growth_rate*5+conv_dim, 1, 0, 1)
        self.tranavg0 = nn.AvgPool1d(2, 2, 0)

        self.db1 = DenseBlock(growth_rate, layer, growth_rate*5+conv_dim, 3, 1, 1)
        self.tranconv1 = BNConv1D(growth_rate*layer*2+conv_dim, growth_rate*layer*2+conv_dim, 1, 0, 1)
        self.tranavg1 = nn.AvgPool1d(2, 2, 0)

        self.db2 = DenseBlock(growth_rate, layer, growth_rate*layer*2+conv_dim, 3, 1, 1)
        self.tranconv2 = BNConv1D(growth_rate*layer*3+conv_dim, growth_rate*layer*3+conv_dim, 1, 0, 1)
        self.tranavg2 = nn.AvgPool1d(2, 2, 0)

        self.db3 = DenseBlock(growth_rate, layer, growth_rate*layer*3+conv_dim, 3, 1, 1)
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = FullyConnect(growth_rate*layer*4+conv_dim, output_class)
    def forward(self, input):
        output=self.conv0(input)
        output=self.maxpool(output)
        output=self.db0(output)
        output=self.tranconv0(output)
        output=self.tranavg0(output)
        output=self.db1(output)
        output=self.tranconv1(output)
        output=self.tranavg1(output)
        output=self.db2(output)
        output=self.tranconv2(output)
        output=self.tranavg2(output)
        output=self.db3(output)
        output=self.avgpool(output)
        output=self.fc(output)
        return output