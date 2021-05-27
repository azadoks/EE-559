#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plotting functions for EE-559 miniproject 2."""

__author__ = "Austin Zadoks"

 
def plot_history(history, filename='history.png'):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(dpi=300, figsize=(7,5))
    
    ax.semilogy(history['train_loss'], label='Train loss', c='tab:blue')
    ax.semilogy(history['test_loss'], label='Test loss', c='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend(loc='upper left')
    
    twin_ax = ax.twinx()
    twin_ax.plot(history['train_acc'], label='Train accuracy', c='tab:red')
    twin_ax.plot(history['test_acc'], label='Test accuracy', c='tab:green')
    twin_ax.set_ylabel('Accuracy [%]')
    twin_ax.legend(loc='upper right')

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
    ax[1].scatter(train_data[0][:,0], train_data[0][:,1], c=prediction.sigmoid().round())
    ax[1].set_title('Train prediction treated')

    prediction = model(test_data[0])
    ax[2].scatter(test_data[0][:,0], test_data[0][:,1], c=prediction)
    ax[2].set_title('Test prediction')
    ax[3].scatter(test_data[0][:,0], test_data[0][:,1], c=prediction.sigmoid().round())
    ax[3].set_title('Test prediction treated')

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    fig.tight_layout()

    fig.savefig(filename)