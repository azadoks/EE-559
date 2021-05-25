#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions for training convolutional neural networks of different architectures
for EE-559 miniproject 1.
"""

__author__ = "Austin Zadoks"

import torch
from torch import nn
from torch.nn import functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)

    # pylint: disable=arguments-differ
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=3)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)
        x = F.relu(self.fc1(x.view(-1, 128)))
        x = self.fc2(x)
        return x
