#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for reporting status and results to the user for EE-559 miniproject 1."""

__author__ = "Austin Zadoks"

import typing as ty

def print_round_header():
    """Print the header for round results."""
    print('Round  |  Train loss  |  Train error  |   Test loss  |  Test errror  |  Time  [s]')
    print('---------------------------------------------------------------------------------')


def print_round_line(result: ty.Dict, r: int, n_rounds: int):
    """Print the results of a round."""
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
    train_loss = results['train_loss']
    test_loss = results['test_loss']
    train_err = results['train_err']
    test_err = results['test_err']
    time = results['time']

    print('Statistics')
    print('----------')
    print(f'Final train loss:         {train_loss.mean():6.2e} +- {train_loss.std():6.2e} [ ]')
    print(f'Final test loss:          {test_loss.mean():6.2e} +- {test_loss.std():6.2e} [ ]')
    print(f'Final train error:        {train_err.mean():8.2f} +- {train_err.std():8.2f} [%]')
    print(f'Final test error:         {test_err.mean():8.2f} +- {test_err.std():8.2f} [%]')
    print(f'Time:                     {time.mean():8.2f} +- {time.std():8.2f} [s]')
    print()