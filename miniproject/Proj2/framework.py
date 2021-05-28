#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mini deep-learning framework for EE-559 miniproject 2."""

__author__ = "Austin Zadoks"

import math

from torch import empty


class Module(object):
    
    def __init__(self):
        return

    def forward(self, *input):
        raise NotImplementedError
    
    def backward(self, *doutput):
        raise NotImplementedError
    
    @property
    def param(self):
        return ()
    
    def zero_grad(self):
        return

    def __call__(self, *input):
        return self.forward(*input)


class Linear(Module):
    """
    A linear layer.

    Weights and biases are initialized using uniform sampling in the range
        (-1/sqrt(in_features), 1/sqrt(in_features))
    so that the variance in the weights and biases is proportional to
        1 / n_features

    w is a matrix with shape (out_features, in_features)
    bias is a vector with length (out_features)

    :param in_features: size of input vector
    :param out_features: size of output vector
    """
    def __init__(self, in_features, out_features):
        self.input = None
        self.in_features = in_features
        self.out_features = out_features
        
        lim = 1 / math.sqrt(self.in_features)  # PyTorch Linear weight initialization

        # initialize weight and bias to uniform sampling (-sqrt(1/features), sqrt(1/features))
        self.weight = empty((self.out_features, self.in_features)).uniform_(-lim, lim)
        self.bias = empty(self.out_features).uniform_(-lim, lim)
        # initialize gradients to 0
        self.dweight = empty((self.out_features, self.in_features)).fill_(0)
        self.dbias = empty(self.out_features).fill_(0)

    def forward(self, input):
        """
        Forward pass.

        y = x @ w.T + b

        :param input: input batch
        :returns: y = x @ w.T + b
        """
        self.input = input
        return self.bias.addmm(self.input, self.weight.t())  # x @ w.T + b

    def backward(self, doutput):
        """
        Backward pass.

        :param doutput: gradient of the previous layer with respect to the output
        :returns: doutput @ weight
        """
        self.dbias.add_(doutput.sum(0))  # dx / db = 
        self.dweight.add_(doutput.t().mm(self.input))
        return doutput.mm(self.weight)

    @property
    def param(self):
        return (self.weight, self.dweight), (self.bias, self.dbias)
 
    def zero_grad(self):
        self.dbias = empty(self.out_features).fill_(0)
        self.dweight = empty(self.out_features, self.in_features).fill_(0)


class Sequential(Module):
    
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
        self.input = None
    
    def forward(self, input):
        self.input = input
        output = input
        for module in self.modules:
            output = module(output)
        return output

    def backward(self, doutput):
        dsequential = doutput
        for module in self.modules[::-1]:
            dsequential = module.backward(dsequential)
        return dsequential

    @property
    def param(self):
        return (module.param for module in self.modules)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()


class ReLU(Module):
    
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return input.relu()

    def backward(self, doutput):
        drelu = self.input.relu()
        drelu[drelu != 0] = 1
        return drelu * doutput


class Tanh(Module):
    
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return input.tanh()

    def backward(self, doutput):
        dtanh = self.input.cosh().pow(-2)
        return dtanh * doutput


class MSELoss(Module):

    def __init__(self, model):
        self.input = None
        self.target = None
        self.prediction = None
        self.model = model

    def forward(self, prediction, target):
        self.target = target
        self.prediction = prediction
        return ((self.prediction - self.target) ** 2).mean()

    def backward(self):
        dloss = 2 * (self.prediction - self.target) / self.prediction.numel()
        self.model.backward(dloss)


class SGD():

    def __init__(self, module, lr=1e-2, momentum=None):
        self.module = module
        self.lr = lr
        self.momentum = momentum
        self.vt = []
        if self.momentum is not None:
            for param_set in self.module.param:
                for _, dparam in param_set:
                    self.vt.append(dparam)

    def step(self):
        for set_idx, param_set in enumerate(self.module.param):
            for param_idx, (param, dparam) in enumerate(param_set):
                if self.momentum is not None:
                    i = set_idx + param_idx
                    self.vt[i] = self.vt[i] * self.momentum + dparam
                    param.sub_(self.lr * self.vt[i])
                else:
                    param.sub_(self.lr * dparam)

