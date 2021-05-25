#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Neural network model classes for EE-559 miniproject 1."""

__author__ = "Austin Zadoks"

import torch
from torch import nn


class ConvNet(nn.Module):
    """
    A simple convolutional neural network.

    1@32x32 -> 2D Convolution (1->32, 3x3, 1)
    -> 32@12x12 -> Avgerage Pooling (1x1, 2)
    -> 32@6x6 -> Tanh
    -> 32@6x6 -> 2D Dropout (0.25)
    -> 32@6x6 -> 2D Convolution (32->64, 3x3, 1)
    -> 64@4x4 -> Average Pooling (1x1, 2)
    -> 64@2x2 -> Tanh
    -> 64@2x2 -> 2D Dropout (0.5)

    """
    def __init__(self) -> None:
        super().__init__()
        self.act = nn.Tanh()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=3,
                               stride=1,
                               padding=0)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=2,
                               stride=1,
                               padding=0)
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=2,
                               stride=1,
                               padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.act(x)

        return x


class FullyConnectedNet(nn.Module):
    """
    A simply fully-connected network.

    256 -> Linear
    -> 128 -> Tanh
    -> 128 -> Linear
    -> 10

    """
    def __init__(self):
        super().__init__()
        self.act = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.do = nn.Dropout(0.25)
        self.lin1 = nn.Linear(in_features=128, out_features=64)
        self.lin2 = nn.Linear(in_features=64, out_features=32)
        self.lin3 = nn.Linear(in_features=32, out_features=10)

    # pylint: disable=arguments-differ
    def forward(self, x):

        x = self.lin1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.do(x)

        x = self.lin2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.do(x)

        x = self.lin3(x)

        return x


class LeNet(nn.Module):
    """
    A LeNet-style convolutional neural network for MNIST character recognition.

    1@14x14 -> ConvNet
    -> 64@2x2 -> Reshape
    -> 256 -> FullyContectedNet
    -> 10

    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = ConvNet()
        self.fc = FullyConnectedNet()

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Proj1Net(nn.Module):
    """
    Use LeNet-style convolutional neural networks to learn if the digit in the
    first MNIST image in a pair is lesser or equal to the digit in the second
    image.

    :param share_weight: Share LeNet weights between first and second image
        networks
    :param aux_loss: Allow auxiliary loss by returning sub-network predictions
    """
    def __init__(self,
                 share_weight: bool = False,
                 aux_loss: bool = False) -> None:
        super().__init__()
        self.share_weight = share_weight
        self.aux_loss = aux_loss

        self.act = nn.Tanh()
        self.bn = nn.BatchNorm1d(10)
        self.do = nn.Dropout(0.25)
        self.lin1 = nn.Linear(20, 10)
        self.lin2 = nn.Linear(10, 2)

        self.lenet1 = LeNet()
        if self.share_weight:
            self.lenet2 = self.lenet1
        else:
            self.lenet2 = LeNet()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x1 = x[:, 0]
        x1 = x1.unsqueeze(1)  # unsqueeze adds back the lost dimension
        x1 = self.lenet1(x1)

        x2 = x[:, 1]
        x2 = x2.unsqueeze(1)
        x2 = self.lenet2(x2)

        x = [x1, x2]
        x = torch.cat(x, dim=1)  # pylint: disable=no-member
        x = self.act(x)

        x = self.lin1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.do(x)

        x = self.lin2(x)

        if self.aux_loss:
            # We need to keep the internal LeNet predictions to calculate the auxiliary loss
            return x, (x1, x2)
        else:
            return x, ()
