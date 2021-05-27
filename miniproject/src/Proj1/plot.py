#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plotting functions for EE-559 miniproject 1"""

__author__ = "Austin Zadoks"

import math

import matplotlib.pyplot as plt
import torch


def plot_histories(histories, filename='history.png'):
    nrow_ncol = math.ceil(math.sqrt(len(histories)))
    fig, axes = plt.subplots(nrow_ncol, dpi=300, figsize=(8*nrow_ncol, 5*nrow_ncol))
    axes= axes.flatten()

    for r, (history, ax) in enumerate(zip(histories, axes)):
        ax.semilogy(history['train_loss'], linewidth=1, label='Training loss', c='tab:blue')
        ax.semilogy(history['test_loss'], linewidth=1, label='Test loss', c='tab:green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(loc='upper left')

        twin_ax = ax.twinx()
        twin_ax.plot(history['train_err'], linewidth=1, label='Training error', c='tab:orange')
        twin_ax.plot(history['test_err'], linewidth=1, label='Test error', c='tab:red')
        twin_ax.set_ylabel('Error')
        twin_ax.legend(loc='upper right')

        ax.set_title(f'Round {r}')

    fig.tight_layout()
    fig.savefig(filename)
