#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main executable for EE-559 miniproject 2."""

__author__ = "Austin Zadoks"

import time

from torch import manual_seed, set_grad_enabled

import data
import torch.nn as framework
import torch.optim as optim
import plot
import train
import log


def train_selected_model(
    activation,
    learning_rate,
    momentum,
    n_points,
    n_epochs,
    batch_size,
    plot_points=False
):
    train_data, test_data = data.generate_data(n_points)

    model = train.build_model(activation)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = framework.MSELoss()

    t0 = time.perf_counter()
    history = train.train_model(model, optimizer, criterion, train_data, test_data, n_epochs,
                                batch_size)
    t1 = time.perf_counter()

    result = {
        'train_loss': train.compute_loss(model, criterion, train_data, batch_size),
        'test_loss': train.compute_error(model, train_data, batch_size) * 100,
        'train_err': train.compute_loss(model, criterion, test_data, batch_size),
        'test_err': train.compute_error(model, test_data, batch_size) * 100,
        'time': t1 - t0
    }

    if plot_points:
        plot.plot_points(test_data, train_data, model, plot_points)

    return history, result


def main(n_rounds, base_seed, plot_history=False, plot_points=False):
    manual_seed(base_seed)

    relu_params = {
        'activation': framework.ReLU,
        'learning_rate': 1e-1,
        'momentum': 0.6,
        'n_points': 1000,
        'n_epochs': 50,
        'batch_size': 64,
        'plot_points': 'relu_points.pdf' if plot_points else False,
    }
    tanh_params = {
        'activation': framework.Tanh,
        'learning_rate': 1e-1,
        'momentum': 0.9,
        'n_points': 1000,
        'n_epochs': 50,
        'batch_size': 64,
        'plot_points': 'tanh_points.pdf' if plot_points else False,
    }

    histories = {'relu': [], 'tanh': []}
    results = {'relu': [], 'tanh': []}

    print('ReLU')
    print(''.join(['=']*100))
    log.print_round_header()
    for r in range(n_rounds):
        history, result = train_selected_model(**relu_params)
        histories['relu'].append(history)
        results['relu'].append(result)
        log.print_round_line(result, r, n_rounds)
    log.print_round_footer()
    log.print_round_statistics(results['relu'])
    if plot_history:
        plot.plot_avg_histories(histories['relu'], 'relu_history.pdf')

    print('tanh')
    print(''.join(['=']*100))
    log.print_round_header()
    for r in range(n_rounds):
        history, result = train_selected_model(**tanh_params)
        histories['tanh'].append(history)
        results['tanh'].append(result)
        log.print_round_line(result, r, n_rounds)
    log.print_round_footer()
    log.print_round_statistics(results['tanh'])
    if plot_history:
        plot.plot_avg_histories(histories['tanh'], 'tanh_history.pdf')


if __name__ == '__main__':
    n_rounds = 32
    base_seed = 2021

    manual_seed(2021)
    main(n_rounds, base_seed, plot_history=True, plot_points=True)
