#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plotting functions for EE-559 miniproject 2."""

__author__ = "Austin Zadoks"

import math
import typing as ty


def plot_avg_histories(all_histories: ty.List[ty.Dict], filename: str='avg_history.png'):
    """
    Plot training histories for a set of rounds.
    
    :param histories: list of history dictionaries
    :param filename: filename for plot saving
    """
    try:
        import matplotlib.pyplot as plt
        from torch import vstack
    except ImportError:
        return

    def compute_history_stats(histories):
        history_statistics = {}
        for history in histories:
            for key, value in history.items():
                if key not in history_statistics:
                    history_statistics[key] = value
                else:
                    history_statistics[key] = vstack(  # pylint: disable=no-member
                        [history_statistics[key], value])
        return {
            key: (value.mean(0), value.std(0))
            for key, value in history_statistics.items()
        }

    history_statistics = compute_history_stats(all_histories)

    fig, ax = plt.subplots(dpi=300, figsize=(7,5))

    x = [i for i in range(history_statistics['train_loss'][0].size(0))]
    
    ax.semilogy(x, 
        history_statistics['train_loss'][0],
        c='black',
        label='Training loss')
    ax.fill_between(x,
        history_statistics['train_loss'][0] - history_statistics['train_loss'][1],
        history_statistics['train_loss'][0] + history_statistics['train_loss'][1],
        alpha=0.2, color='black')
    ax.semilogy(x, 
        history_statistics['test_loss'][0],
        c='grey',
        label='Test loss')
    ax.fill_between(x,
        history_statistics['test_loss'][0] - history_statistics['test_loss'][1],
        history_statistics['test_loss'][0] + history_statistics['test_loss'][1],
        alpha=0.2, color='grey')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE loss')

    twin_ax = ax.twinx()
    twin_ax.plot(x,
        1 - history_statistics['train_err'][0],
        c='tab:purple',
        label='Training acc.')
    twin_ax.fill_between(x,
        (1 - history_statistics['train_err'][0]) - history_statistics['train_err'][0],
        (1 - history_statistics['train_err'][0]) + history_statistics['train_err'][0],
        alpha=0.2, color='tab:purple')
    twin_ax.plot(x,
        1 - history_statistics['test_err'][0],
        c='tab:pink',
        label='Test acc.')
    twin_ax.fill_between(x,
        (1 - history_statistics['test_err'][0]) - history_statistics['test_err'][0],
        (1 - history_statistics['test_err'][0]) + history_statistics['test_err'][0],
        alpha=0.2, color='tab:pink')
    twin_ax.set_ylabel('Accuracy')

    fig.legend(loc='upper center', ncol=4)
    fig.tight_layout()
    fig.savefig(filename)



def plot_points(train_data, test_data, model, filename='points.png'):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(2, 2, dpi=300, figsize=(8,8))
    ax = ax.flatten()

    prediction = model(train_data[0])
    ax[0].scatter(train_data[0][:,0], train_data[0][:,1], c=prediction)
    ax[0].set_title('Train prediction')
    ax[1].scatter(train_data[0][:,0], train_data[0][:,1], c=prediction.tanh().round())
    ax[1].set_title('Train prediction treated')

    prediction = model(test_data[0])
    ax[2].scatter(test_data[0][:,0], test_data[0][:,1], c=prediction)
    ax[2].set_title('Test prediction')
    ax[3].scatter(test_data[0][:,0], test_data[0][:,1], c=prediction.tanh().round())
    ax[3].set_title('Test prediction treated')

    for a in ax:
        circle = plt.Circle((0.5, 0.5), 1 / math.sqrt(2 * math.pi), color='gray', alpha=0.5)
        a.set_aspect(1)
        a.add_artist(circle)

    fig.tight_layout()

    fig.savefig(filename)