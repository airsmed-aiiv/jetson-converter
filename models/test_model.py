'''
File: test_model.py
Project: jetson-converter
File Created: 2023-03-09 16:26:59
Author: sangminlee
-----
This script ...
Reference
...
'''

import torch
from timm.models.layers.activations import Swish


class TestModel(torch.nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        # self.conv2d_1 = torch.nn.Conv2d(1, 4, (3, 3), groups=1)
        # self.conv2d_1.weight = torch.nn.Parameter(torch.ones_like(self.conv2d_1.weight) / 9.)
        # self.conv2d_1.bias = torch.nn.Parameter(torch.zeros_like(self.conv2d_1.bias) / 2.)
        self.swish = Swish()

    def forward(self, x):
        # x = self.conv2d_1(x)
        x = self.swish(x)
        return x
