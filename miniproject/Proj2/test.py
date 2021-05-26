#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main executable for EE-559 miniproject 2."""

__author__ = "Austin Zadoks"

import math

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

    history = train.train_model(model, optimizer, criterion, train_data, test_data, n_epochs,
                                batch_size)

    print(f'Final train loss:       {train.compute_loss(model, criterion, train_data, batch_size):8.4e}')
    print(f'Final train accuracy:     {train.compute_accuracy(model, train_data, batch_size):8.2f}')
    print(f'Final test loss:        {train.compute_loss(model, criterion, test_data, batch_size):8.4e}')
    print(f'Final test accuracy:      {train.compute_accuracy(model, test_data, batch_size):8.2f}')

    if plot_history:
        plot.plot_history(history, plot_history)
    if plot_points:
        plot.plot_points(test_data, train_data, model, plot_points)


def main(plot_history=False, plot_points=False):
    relu_params = {
        'activation': framework.ReLU,
        'learning_rate': 4e-2,
        'momentum': 0.4,
        'plot_history': 'relu_history.png' if plot_history else False,
        'plot_points': 'relu_points.png' if plot_points else False
    }
    tanh_params = {
        'activation': framework.Tanh,
        'learning_rate': 1.8e-1,
        'momentum': 0.5,
        'plot_history': 'tanh_history.png' if plot_history else False,
        'plot_points': 'tanh_points.png' if plot_points else False
    }
    
    print('ReLU')
    print(''.join(['=']*100))
    train_selected_model(**relu_params)

    print('Tanh')
    print(''.join(['=']*100))
    train_selected_model(**tanh_params)


if __name__ == '__main__':
    set_grad_enabled(False)
    manual_seed(2021)
    main(True, True)
