#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plotting functions for EE-559 miniproject 1"""

__author__ = "Austin Zadoks"

import typing as ty

import matplotlib.pyplot as plt


def plot_histories(histories: ty.List[ty.Dict], nrow: int, ncol: int, filename: str='history.png'):
    """
    Plot training histories for a set of rounds.
    
    :param histories: list of history dictionaries
    :param nrow: number of subplot rows
    :param ncol: number of subplot columns
    :param filename: filename for plot saving
    """
    fig, axes = plt.subplots(nrow, ncol, dpi=300, figsize=(8*ncol, 5*nrow))
    axes= axes.flatten()

    for r, (history, ax) in enumerate(zip(histories, axes[:len(histories)])):
        ax.semilogy(history['train_loss'], linewidth=1, label='Training loss', c='tab:blue')
        # ax.semilogy(history['test_loss'], linewidth=1, label='Test loss', c='tab:green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(loc='upper left')

        twin_ax = ax.twinx()
        twin_ax.plot(1 - history['train_err'], linewidth=1, label='Training accuracy', c='tab:orange')
        twin_ax.plot(1 - history['test_err'], linewidth=1, label='Test accuracy', c='tab:red')
        twin_ax.set_ylabel('Accuracy')
        twin_ax.legend(loc='upper right')

        ax.set_title(f'Round {r+1}')
    
    for ax in axes[len(histories):]:
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(filename)
