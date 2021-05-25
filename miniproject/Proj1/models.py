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
                               kernel_size=3,
                               stride=1,
                               padding=0)
        self.drop1 = nn.Dropout2d(p=0.3)
        self.drop2 = nn.Dropout2d(p=0.3)
        self.pool = nn.AvgPool2d(kernel_size=1, stride=2, padding=0)

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Nx1x14x14
        x = self.conv1(x)  # Nx32x12x12
        x = self.act(x)  # Nx32x12x12
        x = self.pool(x)  # Nx32x6x6

        x = self.drop1(x)  # Nx32x6x6

        x = self.conv2(x)  # Nx64x4x4
        x = self.act(x)  # Nx64x4x4
        x = self.pool(x)  # Nx64x2x2

        x = self.drop2(x)  # Nx64x2x2

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
        self.lin1 = nn.Linear(in_features=256, out_features=128)
        self.lin2 = nn.Linear(in_features=128, out_features=10)

    # pylint: disable=arguments-differ
    def forward(self, x):
        # Nx256
        x = self.lin1(x)  # Nx128
        x = self.act(x)  # Nx128

        x = self.lin2(x)  # Nx10

        return x


class LeNet(nn.Module):
    """
    A LeNet-style convolutional neural network for character recognition for
    MNIST.

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
        # Nx1x14x14
        x = self.conv(x)  # Nx64x2x2
        x = x.view(x.size(0), -1)  # Nx256
        x = self.fc(x)  # Nx10

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
        self.lenet1 = LeNet()
        if self.share_weight:
            self.lenet2 = self.lenet1
        else:
            self.lenet2 = LeNet()
        self.act = nn.Tanh()
        self.lin = nn.Linear(20, 2)  # Makes value(img1) <= value(img2) prediciton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Nx2x14x14
        x1 = x[:, 0]  # Nx14x14
        x1 = x1.unsqueeze(
            1)  # Nx1x14x14 (unsqueeze adds back the lost dimension)
        x1 = self.lenet1(x1)

        x2 = x[:, 1]  # Nx14x14
        x2 = x2.unsqueeze(1)  # Nx1x14x14
        x2 = self.lenet2(x2)

        x = [x1, x2]  # [Nx10, Nx10]
        x = torch.cat(x, dim=1)  # Nx20  pylint: disable=no-member
        x = self.act(x)  # Nx20
        x = self.lin(x)  # Nx2

        if self.aux_loss:
            # We need to keep the LeNet predictions to calculate the auxiliary loss
            return x, x1, x2
        else:
            return x, None, None
