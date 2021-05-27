#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main executable for EE-559 miniproject 2."""

__author__ = "Austin Zadoks"

import math
import time

from torch import manual_seed, set_grad_enabled

import data
import framework
import plot
import train


def train_selected_model(
    activation,
    learning_rate,
    momentum,
    plot_history=False,
    plot_points=False
):
    n_points = 1000
    n_epochs = 500
    batch_size = 64

    train_data, test_data = data.generate_data(n_points)

    model = train.build_model(activation)
    optimizer = framework.SGD(model, lr=learning_rate, momentum=momentum)
    criterion = framework.MSELoss(model)

    t0 = time.perf_counter()
    history = train.train_model(model, optimizer, criterion, train_data, test_data, n_epochs,
                                batch_size)
    t1 = time.perf_counter()

    train_loss = train.compute_loss(model, criterion, train_data, batch_size)
    train_acc = train.compute_accuracy(model, train_data, batch_size) * 100
    test_loss = train.compute_loss(model, criterion, test_data, batch_size)
    test_acc = train.compute_accuracy(model, test_data, batch_size) * 100

    print(f'Final train loss:       {train_loss:8.4e} [ ]')
    print(f'Final train accuracy:     {train_acc:8.2f} [%]')
    print(f'Final test loss:        {test_loss:8.4e} [ ]')
    print(f'Final test accuracy:      {test_acc:8.2f} [%]')
    print(f'Time:                     {t1 - t0:8.2f} [s]')

    if plot_history:
        plot.plot_history(history, plot_history)
    if plot_points:
        plot.plot_points(test_data, train_data, model, plot_points)


def main(plot_history=False, plot_points=False):
    relu_params = {
        'activation': framework.ReLU,
        'learning_rate': 4e-2,
        'momentum': 0.4,
        'plot_history': 'relu_history.pdf' if plot_history else False,
        'plot_points': 'relu_points.pdf' if plot_points else False
    }
    tanh_params = {
        'activation': framework.Tanh,
        'learning_rate': 1.8e-1,
        'momentum': 0.5,
        'plot_history': 'tanh_history.pdf' if plot_history else False,
        'plot_points': 'tanh_points.pdf' if plot_points else False
    }
    
    print('ReLU')
    print(''.join(['=']*100))
    train_selected_model(**relu_params)

    print()

    print('Tanh')
    print(''.join(['=']*100))
    train_selected_model(**tanh_params)


if __name__ == '__main__':
    set_grad_enabled(False)
    manual_seed(2021)
    main(True, True)
