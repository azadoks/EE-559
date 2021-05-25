#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plotting functions for EE-559 miniproject 1"""

__author__ = "Austin Zadoks"

import matplotlib.pyplot as plt
import torch


def plot_loss_acc(loss: torch.Tensor, acc: torch.Tensor, title: str):
    fig, ax = plt.subplots(dpi=300)

    loss_ax = ax
    loss_ax.semilogy(loss.mean(1).detach().numpy(), c='tab:blue', label='Train loss')
    loss_ax.set_xlabel('Epoch')
    loss_ax.set_ylabel('Loss')
    loss_ax.set_title(title)

    acc_ax = loss_ax.twinx()
    acc_ax.plot(acc.mean(1).detach().numpy(), c='tab:red', label='Train accuracy')
    acc_ax.set_ylabel('Accuracy [%]')

    fig.tight_layout()
    plt.savefig(f'plots/{title}.pdf')
    