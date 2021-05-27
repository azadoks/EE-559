#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for tranining neural networks for EE-559 miniproject 1."""

__author__ = "Austin Zadoks"

import typing as ty

import torch
from torch.utils import data

import models


def compute_loss(model: models.Proj1Net, criterion, loader: data.DataLoader) -> float:
    """
    Compute average batch loss.
    
    :param model: Proj1Net model
    :param criterion: loss criterion
    :param loader: data loader
    :returns: average batch loss
    """
    loss_acc = 0
    for input, targets in loader:
        predictions = model(input)
        for i in range(len(predictions)):
            loss_acc += criterion(predictions[i], targets[i])
    return loss_acc * loader.batch_size / len(loader.dataset)


def compute_error(model: models.Proj1Net, loader: data.DataLoader) -> float:
    """
    Compute error rate over batches using winner-take-all method.
    
    :param model: Proj1Net model
    :param loader: data loader
    :returns: winner-take-all error rate [0,1]
    """
    n_errors_acc = 0
    for input, targets in loader:
        predictions = model(input)
        n_errors_acc += (predictions[0].argmax(1) != targets[0]).sum()  # winner-take-all
    return n_errors_acc / len(loader.dataset)


def train_proj1net(model: models.Proj1Net,
                   optimizer,
                   criterion,
                   train_loader: data.DataLoader,
                   test_loader: data.DataLoader,
                   n_epochs: int,
                   full_history: bool) -> ty.Dict[str, torch.Tensor]:
    """
    Train a LeNet-style convolutional neural network to learn if the digit in the first MNIST
    image in a pair is lesser or equal to the digit in the second image.

    :param model: Proj1Net model
    :param optimizer: optimizer
    :param criterion: loss criterion
    :param train_loader: training data loader
    :param test_loader: test data loader
    :param n_epochs: number of epochs
    :full_history: track train and test loss and error
    :returns: history dictionary with training and test loss and error data
    """
    history = {
        'train_loss': torch.zeros(n_epochs),  # pylint: disable=no-member
        'test_loss': torch.zeros(n_epochs),  # pylint: disable=no-member
        'train_err': torch.zeros(n_epochs),  # pylint: disable=no-member
        'test_err': torch.zeros(n_epochs)  # pylint: disable=no-member
    }

    model.train(True)
    for epoch in range(n_epochs):
        for train_input, train_targets in train_loader:
            loss = 0
            # forward
            predictions = model(train_input)
            for i in range(len(predictions)):
                loss += criterion(predictions[i], train_targets[i])

            # backward and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track training loss
            with torch.no_grad():
                history['train_loss'][epoch] += (loss)

        # track training and test loss and errors
        model.train(False)
        with torch.no_grad():  # pylint: disable=no-member
            history['train_loss'][epoch] /= (len(train_loader.dataset) / train_loader.batch_size)
            if full_history:
                history['test_loss'][epoch] = compute_loss(model, criterion, test_loader)
                history['train_err'][epoch] = compute_error(model, train_loader)
                history['test_err'][epoch] = compute_error(model, test_loader)
        model.train(True)

    return history
