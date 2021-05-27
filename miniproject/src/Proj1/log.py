#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for reporting status and results to the user for EE-559 miniproject 1."""

__author__ = "Austin Zadoks"

def print_round_result(result, r, n_rounds):
    print(f'Round {r+1:2d}/{n_rounds}')
    print('-----------')
    print(f'Final train loss:       {result["train_loss"]:8.4e} [ ]')
    print(f'Final train error:        {result["test_loss"]:8.2f} [%]')
    print(f'Final test loss:        {result["train_err"]:8.4e} [ ]')
    print(f'Final test error:         {result["test_err"]:8.2f} [%]')
    print(f'Time:                     {result["time"]:8.2f} [s]')
    print()


def print_round_statistics(results):
    train_loss = results['train_loss']
    test_loss = results['test_loss']
    train_err = results['train_err']
    test_err = results['test_err']
    time = results['time']

    print('Statistics')
    print('----------')
    print(f'Final train loss:       {train_loss.mean():8.4e} +- {train_loss.std():8.4e} [ ]')
    print(f'Final test loss:        {test_loss.mean():8.4e} +- {test_loss.std():8.4e} [ ]')
    print(f'Final train error:        {train_err.mean():8.2f} +- {train_err.std():8.2f} [%]')
    print(f'Final test error:         {test_err.mean():8.2f} +- {test_err.std():8.2f} [%]')
    print(f'Time:                     {time.mean():8.2f} +- {time.std():8.2f} [s]')
    print()