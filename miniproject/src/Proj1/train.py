#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for tranining neural networks for EE-559 miniproject 1."""

__author__ = "Austin Zadoks"

import torch

import models


def compute_loss(model, criterion, loader):
    loss_acc = 0
    for input, targets in loader:
        predictions = model(input)
        for i in range(len(predictions)):
            loss_acc += criterion(predictions[i], targets[i])
    return loss_acc * loader.batch_size / len(loader.dataset)


def compute_error(model, loader):
    n_errors_acc = 0
    for input, targets in loader:
        predictions = model(input)
        n_errors_acc += (predictions[0].argmax(1) != targets[0]).sum()  # winner-take-all
    return n_errors_acc / len(loader.dataset)


def train_proj1net(model: models.Proj1Net, optimizer, criterion, train_loader, test_loader,
                   n_epochs):
    """
    Train a LeNet-style convolutional neural network to learn if the digit in the first MNIST
    image in a pair is lesser or equal to the digit in the second image.
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
            history['test_loss'][epoch] = compute_loss(model, criterion, test_loader)
            history['train_err'][epoch] = compute_error(model, train_loader)
            history['test_err'][epoch] = compute_error(model, test_loader)
        model.train(True)

    return history
