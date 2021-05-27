#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main executable for EE-559 miniproject 1."""

__author__ = "Austin Zadoks"

import time
import typing as ty

import torch
from torch import nn, optim

import data
import log
import models
import plot
import train


def train_round(criterion,
                optimizer,
                optimizer_params: ty.Dict,
                share_weight: bool,
                aux_loss: bool,
                n_pairs: int,
                batch_size: int,
                n_epochs: int,
                full_history: bool) -> ty.Tuple[ty.Dict]:
    """
    Perform a round of training on a Proj1Net model.

    :param criterion: loss criterion
    :param optimizer: optimizer
    :param optimizer_params: optimizer parameters dictionary
    :param share_weight: share subnet weights within MNIST image pairs
    :param aux_loss: use auxiliary loss on subnet outputs
    :n_pairs: number of image pairs
    :batch_size: number of image pairs per batch
    :n_epochs: number of epochs for training
    :full_history: track train and test loss and error
    :returns: training history dictionary and final loss, error, and time dictionary
    """
    train_loader, test_loader = data.load_data(n_pairs, batch_size)
    model = models.Proj1Net(share_weight, aux_loss)
    criterion = criterion()
    optimizer = optimizer(model.parameters(), **optimizer_params)

    t0 = time.perf_counter()
    history = train.train_proj1net(model, optimizer, criterion, train_loader, test_loader, n_epochs,
                                   full_history=full_history)
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


def train_n_rounds(n_rounds: int,
                   base_seed: ty.Optional[int]=None,
                   do_plot: bool=False,
                   full_history: bool=False,
                   **kwargs) -> ty.List[ty.Dict]:
    """
    Train a Proj1Net model for many rounds with different random seeds. Prints round results and
    mean +- standard devation over all rounds.

    :param n_rounds: number of rounds
    :param base_seed: starting seed, incremeted by 1 each round
    :param do_plot: create histories plot after training
    :param full_history: track train and test loss and error
    :param kwargs: keyword arguments passed to `train_round`
    :returns: results for each round
    """
    results = {
        'train_loss': torch.zeros(n_rounds),  # pylint: disable=no-member
        'test_loss': torch.zeros(n_rounds),  # pylint: disable=no-member
        'train_err': torch.zeros(n_rounds),  # pylint: disable=no-member
        'test_err': torch.zeros(n_rounds),  # pylint: disable=no-member
        'time': torch.zeros(n_rounds)  # pylint: disable=no-member
    }
    histories = []

    log.print_round_header()
    for r in range(n_rounds):
        if base_seed is not None:
            torch.manual_seed(base_seed + r)

        history, result = train_round(full_history=full_history, **kwargs)
        
        histories.append(history)
        for key, value in result.items():
            results[key][r] = value
        
        log.print_round_line(result, r, n_rounds)
    log.print_round_footer()

    log.print_round_statistics(results)
    if do_plot:
        plot.plot_histories(
            histories,
            nrow=4,
            ncol=4,
            filename=(
                f'plots/share_weight={kwargs["share_weight"]}__aux_loss={kwargs["aux_loss"]}.png'
            )
        )

    return results


def main():
    """
    Train a Proj1Net model for combinations of weight sharing and auxiliary loss for multiple
    rounds.
    """
    seed = 2021
    n_rounds = 16
    plot = False
    full_history = False

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
        print()
        
        training_parameters['share_weight'] = share_weight
        training_parameters['aux_loss'] = aux_loss
        train_n_rounds(n_rounds, base_seed=seed, do_plot=plot, full_history=full_history,
                       **training_parameters)
        
        print()


if __name__ == '__main__':
    main()