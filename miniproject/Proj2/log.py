#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for reporting status and results to the user for EE-559 miniproject 1."""

__author__ = "Austin Zadoks"

import typing as ty

from torch import empty  # pylint: disable=no-name-in-module

def print_round_header():
    """Print the header for round results."""
    print('Round  |  Train loss  |  Train error  |   Test loss  |  Test errror  |  Time  [s]')
    print('---------------------------------------------------------------------------------')


def print_round_line(result: dict, r: int, n_rounds: int):
    """
    Print the results of a round.
    
    :param results: dictionary
    :param r: round index
    :param n_rounds: number of rounds
    """
    end = '  |  '
    print(f'{r+1:2d}/{n_rounds:2d}', end=end)
    print(f'{result["train_loss"]:8.4e}', end=end)
    print(f'{result["train_err"]:11.2f}', end=end)
    print(f'{result["test_loss"]:8.4e}', end=end)
    print(f'{result["test_err"]:11.2f}', end=end)
    print(f'{result["time"]:9.2f}')


def print_round_footer():
    """Print the footer for round results."""
    print('---------------------------------------------------------------------------------')
    print()


def print_round_statistics(results: ty.List[ty.Dict]):
    """
    Print training statistics for a set of rounds.

    :param results: list of result dictionaries
    """
    n_results = len(results)
    tensor_results = {key: empty(n_results) for key in results[0]}
    for r, result in enumerate(results):
        for key, value in result.items():
            tensor_results[key][r] = value

    train_loss = tensor_results['train_loss']
    test_loss = tensor_results['test_loss']
    train_err = tensor_results['train_err']
    test_err = tensor_results['test_err']
    time = tensor_results['time']

    print('Statistics')
    print('----------')
    print(f'Final train loss:         {train_loss.mean():6.2e} +- {train_loss.std():6.2e} [ ]')
    print(f'Final test loss:          {test_loss.mean():6.2e} +- {test_loss.std():6.2e} [ ]')
    print(f'Final train error:        {train_err.mean():8.2f} +- {train_err.std():8.2f} [%]')
    print(f'Final test error:         {test_err.mean():8.2f} +- {test_err.std():8.2f} [%]')
    print(f'Time:                     {time.mean():8.2f} +- {time.std():8.2f} [s]')
    print()