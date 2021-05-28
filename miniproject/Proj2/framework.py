#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mini deep-learning framework for EE-559 miniproject 2."""

__author__ = "Austin Zadoks"

import math

from torch import empty  # pylint: disable=no-name-in-module


class Module(object):
    """Base class for mini deep-learning framework modules."""
    def __init__(self):
        return

    def forward(self, *input):
        """Perform forward pass."""
        raise NotImplementedError
    
    def backward(self, *doutput):
        """Perform backward pass."""
        raise NotImplementedError
    
    @property
    def param(self):
        """Return (parameter, gradient) tuples."""
        return ()
    
    def zero_grad(self):
        """Zero all stored gradients."""
        return

    def __call__(self, *input):
        """Magic method so that `Module(input)` runs a forward pass."""
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
    def __init__(self, in_features: int, out_features: int):
        self.input = None
        self.in_features = in_features
        self.out_features = out_features
        
        lim = 1 / math.sqrt(self.in_features)  # PyTorch Linear weight initialization

        # note: weight and bias are transposed w.r.t. linear algebra representation because
        # PyTorch Tensors are row-major
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
        return self.bias.addmm(self.input, self.weight.t())

    def backward(self, doutput):
        """
        Backward pass.

        :param doutput: gradient of the previous layer with respect to the output
        :returns: doutput @ weight
        """
        self.dbias.add_(doutput.sum(0))
        self.dweight.add_(doutput.t().mm(self.input))
        return doutput.mm(self.weight)

    @property
    def param(self):
        """
        Model parameters.
        
        :returns: (weight, weight gradient), (bias, bias gradient)
        """
        return (self.weight, self.dweight), (self.bias, self.dbias)
 
    def zero_grad(self):
        """Zero weight and bias gradients."""
        self.dbias = empty(self.out_features).fill_(0)
        self.dweight = empty(self.out_features, self.in_features).fill_(0)


class Sequential(Module):
    """
    A sequential model.

    :param *modules: modules in sequential order
    """
    def __init__(self, *modules: Module):
        super().__init__()
        self.modules = modules
        self.input = None
    
    def forward(self, input):
        """
        Forward pass.

        Input is passed sequentially through all submodules.

        :param input: input batch
        :returns: output of the final submodule
        """
        self.input = input
        output = input
        for module in self.modules:
            output = module(output)
        return output

    def backward(self, doutput):
        """
        Backward pass.

        Gradient is passed reverse-sequentially through all submodules.

        :param input: gradient of loss
        :returns: gradient of the loss w.r.t the first module
        """
        dsequential = doutput
        for module in self.modules[::-1]:
            dsequential = module.backward(dsequential)
        return dsequential

    @property
    def param(self):
        """
        All submodule parameters.

        :returns: nested tuple of submodule parameters
        """
        return (module.param for module in self.modules)

    def zero_grad(self):
        """Zero the gradients of all submodules."""
        for module in self.modules:
            module.zero_grad()


class ReLU(Module):
    """Rectified linear unit activation."""
    def __init__(self):
        self.input = None

    def forward(self, input):
        """
        Forward pass.

        :param input: input batch
        :returns: f(x) = 0 where x < 0 elsewhere x
        """
        self.input = input
        return input.relu()

    def backward(self, doutput):
        """
        Backward pass.

        :param doutput: dl/dx
        :returns: dl/dx * ReLU'(s)
        """
        drelu = self.input.relu()
        drelu[drelu != 0] = 1
        return drelu * doutput


class Tanh(Module):
    """Hyperbolic tangent activation."""
    def __init__(self):
        self.input = None

    def forward(self, input):
        """
        Forward pass.

        :param input: input batch
        :returns: tanh(x)
        """
        self.input = input
        return input.tanh()

    def backward(self, doutput):
        """
        Backward pass.

        :param doutput: dl/dx
        :returns: dl/dx * tanh'(s)
        """
        dtanh = self.input.cosh().pow(-2)
        return dtanh * doutput


class MSELoss(Module):
    """
    Mean-squared error loss.
    
    :param model: model
    """
    def __init__(self, model: Module):
        self.input = None
        self.target = None
        self.prediction = None
        self.model = model

    def forward(self, prediction, target):
        """
        Forward pass.

        :param prediction: model output
        :param target: target output
        :returns: 1/N||prediction - target||_2^2
        """
        self.target = target
        self.prediction = prediction
        return ((self.prediction - self.target) ** 2).mean()

    def backward(self):
        """
        Backward pass.

        Calculates the gradient of loss with respect to the prediction as
            dl/dprediction = 2/N(prediction - target)
        and passes this to the model backward method.
        """
        dloss = 2 * (self.prediction - self.target) / self.prediction.numel()
        self.model.backward(dloss)


class SGD():
    """
    Stochastic gradient descent.

    :param lr: learning rate (eta)
    :param momentum: optional momentum (alpha)
    """
    def __init__(self, module, lr=1e-2, momentum=0):
        self.module = module
        self.lr = lr
        self.momentum = momentum
        self.velocity = []
        # initialize velocity for each parameter as (-eta * dparameter) because
        # the "previous" velocity is 0
        for param_set in self.module.param:
            for _, dparam in param_set:
                self.velocity.append(-self.lr * dparam)

    def step(self):
        """Perform an optimizer step."""
        for set_idx, param_set in enumerate(self.module.param):
            for param_idx, (param, dparam) in enumerate(param_set):
                # update the velocity
                i = set_idx + param_idx
                self.velocity[i] = self.momentum * self.velocity[i] - self.lr * dparam
                # update the parameter
                param.add_(self.velocity[i])

    def zero_grad(self):
        """Zero the module gradients."""
        self.module.zero_grad()
