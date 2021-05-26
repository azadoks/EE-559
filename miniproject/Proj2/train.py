#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model construction and training for EE-559 miniproject 2."""

__author__ = "Austin Zadoks"

from torch import empty

import framework


def build_model(act):
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


def compute_accuracy(model, data, batch_size):
    acc = []
    for batch in zip(data[0].split(batch_size), data[1].split(batch_size)):
        prediction = model(batch[0])
        acc.append((prediction.sigmoid().round() == batch[1]).sum() / batch[1].numel())
    return sum(acc) / len(acc)


def compute_loss(model, criterion, data, batch_size):
    loss = []
    for batch in zip(data[0].split(batch_size), data[1].split(batch_size)):
        prediction = model(batch[0])
        loss.append(criterion(prediction, batch[1]))
    return sum(loss) / len(loss)


def train_model(model, optimizer, criterion, train_data, test_data, n_epochs, batch_size):
    history = {
        'train_loss': empty(n_epochs).fill_(0),
        'test_loss': empty(n_epochs).fill_(0),
        'train_acc': empty(n_epochs).fill_(0),
        'test_acc': empty(n_epochs).fill_(0)
    }

    for epoch in range(n_epochs):
        for train_batch in zip(train_data[0].split(batch_size), train_data[1].split(batch_size)):
            prediction = model(train_batch[0])
            loss = criterion(prediction, train_batch[1])
            history['train_loss'][epoch] += loss

            model.zero_grad()
            criterion.backward()
            optimizer.step()
        
        history['train_loss'][epoch] = history['train_loss'][epoch] / batch_size
        history['train_acc'][epoch] = compute_accuracy(model, train_data, batch_size)
        history['test_loss'][epoch] = compute_loss(model, criterion, test_data, batch_size)
        history['test_acc'][epoch] = compute_accuracy(model, test_data, batch_size)

    return history