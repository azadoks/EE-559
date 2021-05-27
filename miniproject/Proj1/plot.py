#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plotting functions for EE-559 miniproject 1"""

__author__ = "Austin Zadoks"

import typing as ty

import matplotlib.pyplot as plt
import torch


def plot_avg_histories(
    all_histories: ty.Dict[ty.Tuple[bool], ty.List[ty.Dict]],
    filename: str='avg_history.png'
):
    """
    Plot training histories for a set of rounds.
    
    :param histories: list of history dictionaries
    :param nrow: number of subplot rows
    :param ncol: number of subplot columns
    :param filename: filename for plot saving
    """
    def compute_history_stats(histories):
        history_statistics = {}
        for history in histories:
            for key, value in history.items():
                if key not in history_statistics:
                    history_statistics[key] = value
                else:
                    history_statistics[key] = torch.vstack(  # pylint: disable=no-member
                        [history_statistics[key], value])
        return {
            key: (value.mean(0), value.std(0))
            for key, value in history_statistics.items()
        }

    history_statistics = {
        key: compute_history_stats(value)
        for key, value in all_histories.items()
    }

    fig, axes = plt.subplots(2, 2, dpi=300, figsize=(10, 6))
    axes = axes.flatten()

    for i, (history_data, ax) in enumerate(zip(history_statistics.items(), axes)):
        (share_weight, aux_loss), history = history_data
        x = torch.arange(history['train_loss'][0].size(0))  # pylint: disable=no-member
        
        ax.semilogy(x, 
            history['train_loss'][0],
            c='tab:blue',
            label='Training loss' if i == 0 else None)
        ax.fill_between(x,
            history['train_loss'][0] - history['train_loss'][1],
            history['train_loss'][0] + history['train_loss'][1],
            alpha=0.2, color='tab:blue')
        ax.semilogy(x, 
            history['test_loss'][0],
            c='tab:green',
            label='Test loss' if i == 0 else None)
        ax.fill_between(x,
            history['test_loss'][0] - history['test_loss'][1],
            history['test_loss'][0] + history['test_loss'][1],
            alpha=0.2, color='tab:green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

        twin_ax = ax.twinx()
        twin_ax.plot(x,
            1 - history['train_err'][0],
            c='tab:orange',
            label='Training acc.' if i == 0 else None)
        twin_ax.fill_between(x,
            (1 - history['train_err'][0]) - history['train_err'][0],
            (1 - history['train_err'][0]) + history['train_err'][0],
            alpha=0.2, color='tab:orange')
        twin_ax.plot(x,
            1 - history['test_err'][0],
            c='tab:red',
            label='Test acc.' if i == 0 else None)
        twin_ax.fill_between(x,
            (1 - history['test_err'][0]) - history['test_err'][0],
            (1 - history['test_err'][0]) + history['test_err'][0],
            alpha=0.2, color='tab:red')
        twin_ax.set_ylabel('Accuracy')

        ax.set_title(f'Wt. share={share_weight}, Aux. loss={aux_loss}')

    fig.legend(loc='lower center', ncol=2)
    fig.tight_layout()
    fig.savefig(filename)
