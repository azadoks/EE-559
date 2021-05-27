#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Neural network model classes for EE-559 miniproject 1."""

__author__ = "Austin Zadoks"

import torch
from torch import nn


class LeNet(nn.Module):
    """
    A LeNet-style convolutional neural network for MNIST character recognition.
    """
    def __init__(self, conv_out1=32, conv_out2=64, lin_out1=32, do1=0.25) -> None:
        super().__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(1, conv_out1, 3)
        self.conv2 = nn.Conv2d(conv_out1, conv_out2, 3)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.lin1 = nn.Linear(conv_out2 * 4, lin_out1)
        self.bn1 = nn.BatchNorm1d(lin_out1)
        self.do1 = nn.Dropout(do1)
        self.lin2 = nn.Linear(lin_out1, 10)


    # pylint: disable=arguments-differ
    def forward(self, input):
        output = self.conv1(input)  # convolution
        output = self.act(output)
        output = self.pool1(output)  # pooling
        output = self.conv2(output)  # convolution
        output = self.act(output)
        output = self.pool1(output)  # pooling
        output = output.view(output.size(0), -1)  # flatten for linear layers
        output = self.lin1(output)  # fully-connected hidden
        output = self.bn1(output)  # batch normalization
        output = self.act(output)
        output = self.do1(output)  # dropout
        output = self.lin2(output)  # fully-connected output

        return output


class Proj1Net(nn.Module):
    """
    Use LeNet-style convolutional neural networks to learn if the digit in the
    first MNIST image in a pair is lesser or equal to the digit in the second
    image.

    :param share_weight: Share LeNet weights between first and second image
        networks
    :param aux_loss: Allow auxiliary loss by returning sub-network predictions
    """
    def __init__(self, share_weight: bool = False, aux_loss: bool = False):
        super().__init__()
        self.share_weight = share_weight
        self.aux_loss = aux_loss

        self.lenet1 = LeNet()
        if self.share_weight:
            self.lenet2 = self.lenet1
        else:
            self.lenet2 = LeNet()

        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(10)
        self.do1 = nn.Dropout(0.25)
        self.lin1 = nn.Linear(20, 10)
        self.lin2 = nn.Linear(10, 2)

    def forward(self, input) :

        input1 = input[:, 0]
        input1 = input1.unsqueeze(1)  # unsqueeze adds back the lost dimension
        output1 = self.lenet1(input1)

        input2 = input[:, 1]
        input2 = input2.unsqueeze(1)
        output2 = self.lenet2(input2)

        output = [output1, output2]
        output = torch.cat(output, dim=1)  # pylint: disable=no-member
        output = self.act(output)

        output = self.lin1(output)  # fully-connected hidden
        output = self.bn1(output)  # batch normalization
        output = self.act(output)
        output = self.do1(output)  # dropout

        output = self.lin2(output)  # fully-connected output

        if self.aux_loss:
            # We need to keep the internal LeNet predictions to calculate the auxiliary loss
            return (output, output1, output2)
        else:
            return (output,)
