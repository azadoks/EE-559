#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model construction and training for EE-559 miniproject 2."""

__author__ = "Austin Zadoks"

from torch import empty  # pylint: disable=no-name-in-module

import framework


def build_model(act):
    """
    Build the sequential model defined in the project prompt.

    :param act: activation module
    """
    return framework.Sequential(
        framework.Linear(2, 25),
        act(),
        framework.Linear(25, 25),
        act(),
        framework.Linear(25, 25),
        act(),
        framework.Linear(25, 25),
        act(),
        framework.Linear(25, 1)
    )


def compute_error(model: framework.Module, data, batch_size: int):
    """
    Compute error rate using minibatches.

    Use a tanh function and rounding to convert model outputs to binary values for computing
    error. The tanh function maps values onto a range between 0 and 1. Rounding its output
    gives either 1 (in the circle) or 0 (outside the circle).

    :param model: model
    :param data: (input, target)
    :param batch_size: batch size
    :returns: error rate
    """
    err = []
    for batch in zip(data[0].split(batch_size), data[1].split(batch_size)):
        prediction = model(batch[0])
        err.append((prediction.tanh().round() != batch[1]).sum() / batch[1].numel())
    return sum(err) / len(err)


def compute_loss(model: framework.Module, criterion: framework.Module, data: tuple,
                 batch_size: int):
    """
    Compute loss using minibatches.

    :param model: model
    :param criterion: loss criterion
    :param data: (input, target)
    :param batch_size: batch size
    """
    loss = []
    for batch in zip(data[0].split(batch_size), data[1].split(batch_size)):
        prediction = model(batch[0])
        loss.append(criterion(prediction, batch[1]))
    return sum(loss) / len(loss)


def train_model(model, optimizer, criterion, train_data, test_data, n_epochs, batch_size,
                track_history):
    """
    Train a model using minibatches.

    :param model: model
    :param optimizeer: optimizer
    :param criterion: loss criterion
    :param train_data: (train input, train target)
    :param test_data: (test data, test_target)
    :param n_epochs: number of epochs
    :param batch_size: batch size
    :param track_history: trach loss and error by epoch
    :returns: history dictionary
    """
    history = {
        'train_loss': empty(n_epochs).fill_(0),
        'test_loss': empty(n_epochs).fill_(0),
        'train_err': empty(n_epochs).fill_(0),
        'test_err': empty(n_epochs).fill_(0)
    }

    for epoch in range(n_epochs):
        for train_batch in zip(train_data[0].split(batch_size), train_data[1].split(batch_size)):
            prediction = model(train_batch[0])
            loss = criterion(prediction, train_batch[1])
            if track_history:
                history['train_loss'][epoch] += loss

            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()
        
        if track_history:
            history['train_loss'][epoch] = history['train_loss'][epoch] / batch_size
            history['test_loss'][epoch] = compute_loss(model, criterion, test_data, batch_size)
            history['train_err'][epoch] = compute_error(model, train_data, batch_size)
            history['test_err'][epoch] = compute_error(model, test_data, batch_size)

    return history
