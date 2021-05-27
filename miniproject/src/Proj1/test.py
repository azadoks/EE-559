#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main executable for EE-559 miniproject 1."""

__author__ = "Austin Zadoks"

import time
import warnings

import torch
from torch import nn, optim

import data
import log
import models
import plot
import train


warnings.filterwarnings('ignore')
torch.set_num_threads(1)  # Ensure a fair environment for timing  pylint: disable=no-member


def train_round(
    criterion,
    optimizer,
    optimizer_params,
    share_weight,
    aux_loss,
    n_pairs,
    batch_size,
    n_epochs
):
    train_loader, test_loader = data.load_data(n_pairs, batch_size)
    model = models.Proj1Net(share_weight, aux_loss)
    criterion = criterion()
    optimizer = optimizer(model.parameters(), **optimizer_params)

    t0 = time.perf_counter()
    history = train.train_proj1net(model, optimizer, criterion, train_loader, test_loader, n_epochs)
    t1 = time.perf_counter()

    model.train(False)
    with torch.no_grad():  # pylint: disable=no-member
        result = {
            'train_loss': train.compute_loss(model, criterion, train_loader),
            'test_loss': train.compute_loss(model, criterion, test_loader),
            'train_err': train.compute_error(model, train_loader) * 100,
            'test_err': train.compute_error(model, test_loader) * 100,
            'time': t1 - t0
        }

    return history, result


def train_n_rounds(n_rounds, base_seed=None, do_plot=False, **kwargs):
    results = {
        'train_loss': torch.zeros(n_rounds),  # pylint: disable=no-member
        'test_loss': torch.zeros(n_rounds),  # pylint: disable=no-member
        'train_err': torch.zeros(n_rounds),  # pylint: disable=no-member
        'test_err': torch.zeros(n_rounds),  # pylint: disable=no-member
        'time': torch.zeros(n_rounds)  # pylint: disable=no-member
    }
    histories = []

    for r in range(n_rounds):
        if base_seed is not None:
            torch.manual_seed(base_seed + r)

        history, result = train_round(**kwargs)
        
        histories.append(history)
        for key, value in result.items():
            results[key][r] = value
        
        log.print_round_result(result, r, n_rounds)

    log.print_round_statistics(results)
    if do_plot:
        plot.plot_histories(
            histories,
            filename=(f'round={r}__share_weight={kwargs["share_weight"]}'
                        f'__aux_loss={kwargs["aux_loss"]}.png')
        )

    return results


def main():
    seed = 2021
    n_rounds = 10
    plot = True

    import torchsummary
    torchsummary.summary(models.Proj1Net(), (2, 14, 14))

    training_parameters = {
        'criterion': nn.CrossEntropyLoss,
        'optimizer': optim.Adam,
        'optimizer_params': {'lr': 1e-2},
        'n_pairs': 1000,
        'batch_size': 128,
        'n_epochs': 25
    }

    for share_weight, aux_loss in [(False, False), (True, False), (False, True), (True, True)]:
        print(f'Weight sharing {share_weight}, Auxiliary loss {aux_loss}')
        print(''.join(['=']*100))
        
        training_parameters['share_weight'] = share_weight
        training_parameters['aux_loss'] = aux_loss
        train_n_rounds(n_rounds, base_seed=seed, do_plot=plot, **training_parameters)
        
        print()


if __name__ == '__main__':
    main()
