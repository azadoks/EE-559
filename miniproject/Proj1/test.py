#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main executable for EE-559 miniproject 1."""

__author__ = "Austin Zadoks"

from math import ceil, sqrt
import time

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
import torch
from torch import optim
from torch import nn

import architectures
import dlc_practical_prologue as prologue


def print_block(block: str, title: str = 'BLOCK'):
    N_COLS = 80

    n_hashes = N_COLS - (len(title) + 2)
    hashes = ''.join(['#'] * (n_hashes // 2))
    block = f'{hashes} {title} {hashes}\n' + block + '\n'

    print(block)


# pylint: disable=too-many-arguments
def data_report(trn_inp: torch.Tensor, trn_out: torch.Tensor,
                trn_cls: torch.Tensor, tst_inp: torch.Tensor,
                tst_out: torch.Tensor, tst_cls: torch.Tensor):
    header = {
        'title': 'Name',
        'dimension': 'Tensor dimenision',
        'dtype': 'Type',
        'content': 'Content'
    }
    row_format = '{title:14s} {dimension:24s} {dtype:16s} {content:32s}\n'

    data = [{
        'title': 'Train input',
        'dimension': str(tuple(trn_inp.size())),
        'dtype': str(trn_inp.dtype),
        'content': 'Images'
    }, {
        'title': 'Train target',
        'dimension': str(tuple(trn_out.size())),
        'dtype': str(trn_out.dtype),
        'content': 'Classes to predict ∈ {0,1}'
    }, {
        'title': 'Train classes',
        'dimension': str(tuple(trn_cls.size())),
        'dtype': str(trn_cls.dtype),
        'content': 'Classes of the two digits ∈ {0,...,9}'
    }, {
        'title': 'Test input',
        'dimension': str(tuple(tst_inp.size())),
        'dtype': str(tst_inp.dtype),
        'content': 'Images'
    }, {
        'title': 'Test target',
        'dimension': str(tuple(tst_out.size())),
        'dtype': str(tst_out.dtype),
        'content': 'Classes to predict ∈ {0,1}'
    }, {
        'title': 'Test classes',
        'dimension': str(tuple(tst_cls.size())),
        'dtype': str(tst_cls.dtype),
        'content': 'Classes of the two digits ∈ {0,...,9}'
    }]

    block = row_format.format(**header)
    block += ''.join(['-'] * len(block)) + '\n'
    for data_desc in data:
        block += row_format.format(**data_desc)
    print_block(block, 'DATA INFORMATION')


def train_report(title: str, model, n_rounds: int, n_epochs: int,
                 trn_err: torch.Tensor, tst_err: torch.Tensor,
                 trn_time: torch.Tensor, trn_loss: list, tst_acc: list):
    block = f'Trained {model} in {n_rounds} rounds for {n_epochs} epochs:\n'
    block += f'\tTrain error:      {trn_err.mean():8.4f} +- {trn_err.std():8.4f}\n'
    block += f'\tTest error:       {tst_err.mean():8.4f} +- {tst_err.std():8.4f}\n'
    block += f'\tTrain time:       {trn_time.mean():8.4f} +- {trn_time.std():8.4f} [s]\n'
    block += f'\tTotal train time: {trn_time.sum():8.4f} [s]\n'
    if PLOTTING_AVAILABLE:
        n_row_col = ceil(sqrt(n_rounds))
        fig, ax = plt.subplots(n_row_col,
                               n_row_col,
                               dpi=300,
                               figsize=(6 * n_row_col, 4 * n_row_col))
        ax = ax.flatten()
        for r in range(n_rounds):
            ax[r].semilogy(trn_loss[r].detach().numpy(),
                           label='Train loss',
                           c='tab:blue')
            ax2r = ax[r].twinx()
            ax2r.plot(tst_acc[r].detach().numpy(),
                      label='Test accuracy',
                      c='tab:red')
            ax[r].set_xlabel('Epoch')
            ax[r].set_ylabel('Loss')
            ax2r.set_ylabel('Accuracy')
        fig.suptitle(title)
        fig.tight_layout()
        plt.savefig(f'{title}.pdf')
        print(f'Saved training curve plots to {title}.pdf')
    print_block(block, title)


def train_model(model, trn_inp: torch.Tensor, trn_out: torch.Tensor,
                tst_inp: torch.Tensor, tst_out: torch.Tensor, criterion,
                optimizer, s_mbatch: int, n_epochs: int):
    """
    Train a PyTorch model using minibatches.

    :param model: Model to train
    :param trn_inp: Train inputs
    :param trn_out: Train targets / outputs
    :param criterion: Loss criterion
    :param optimizer: Optimizer
    :param s_mbatch: Minibatch size
    :param n_epochs: Number of epochs to run
    """
    trn_loss = torch.empty((n_epochs, ), dtype=torch.float)  # pylint: disable=no-member
    tst_acc = torch.empty((n_epochs, ), dtype=torch.float)  # pylint: disable=no-member
    for e in range(n_epochs):
        for b in range(0, trn_inp.size(0), s_mbatch):
            trn_inp_batch = trn_inp.narrow(0, b, s_mbatch)
            trn_out_batch = trn_out.narrow(0, b, s_mbatch)

            pred_out = model(trn_inp_batch)
            loss = criterion(pred_out, trn_out_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred_out = model(trn_inp)
        trn_loss[e] = criterion(pred_out, trn_out)
        tst_acc[e] = 1 - compute_n_errors(model, tst_inp, tst_out,
                                          s_mbatch) / tst_inp.size(0)

    return trn_loss, tst_acc


def compute_n_errors(model, inp, out, s_mbatch):
    n_errors = 0

    for b in range(0, inp.size(0), s_mbatch):
        inp_batch = inp.narrow(0, b, s_mbatch)
        out_batch = out.narrow(0, b, s_mbatch)

        pred_out = model(inp_batch)
        _, pred_cls = pred_out.max(1)
        n_errors += (out_batch != pred_cls).sum()

    return n_errors


def simple_conv_net(trn_inp: torch.Tensor, trn_out: torch.Tensor,
                    tst_inp: torch.Tensor, tst_out: torch.Tensor,
                    n_rounds: int, n_epochs: int, s_mbatch: int, lr: float):
    """Train a simple convolutional neural network."""
    n_pairs = trn_inp.size(0)

    tst_n_err = torch.empty((n_rounds, ), dtype=torch.int)  # pylint: disable=no-member
    trn_n_err = torch.empty((n_rounds, ), dtype=torch.int)  # pylint: disable=no-member
    trn_time = torch.empty((n_rounds, ), dtype=torch.float)  # pylint: disable=no-member
    trn_loss = []
    tst_acc = []
    for r in range(n_rounds):
        t0 = time.perf_counter()

        model = architectures.ConvNet()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        plot_data = train_model(model, trn_inp, trn_out, tst_inp, tst_out,
                                criterion, optimizer, s_mbatch, n_epochs)
        trn_loss.append(plot_data[0])
        tst_acc.append(plot_data[1])
        trn_n_err[r] = compute_n_errors(model, trn_inp, trn_out, s_mbatch)
        tst_n_err[r] = compute_n_errors(model, tst_inp, tst_out, s_mbatch)

        t1 = time.perf_counter()
        trn_time[r] = t1 - t0

    trn_err = trn_n_err / (n_pairs)
    tst_err = tst_n_err / (n_pairs)

    return model, trn_err, tst_err, trn_time, trn_loss, tst_acc


def main():
    N_PAIRS = 1000  # Number of pairs, specified by miniproject prompt
    N_ROUNDS = 9  # Number of rounds to train each model

    LR = 1e-3  # Learning rate
    N_EPOCHS = 25  # Number of epochs
    S_MBATCH = 100  # Minibatch size

    # TODO: ensure random seed for each round

    # Load data
    trn_inp, trn_out, trn_cls, tst_inp, tst_out, tst_cls = prologue.generate_pair_sets(
        N_PAIRS)
    data_report(trn_inp, trn_out, trn_cls, tst_inp, tst_out, tst_cls)

    # Simple convolutional neural network
    model, trn_err, tst_err, trn_time, trn_loss, tst_acc = simple_conv_net(
        trn_inp,
        trn_out,
        tst_inp,
        tst_out,
        n_rounds=N_ROUNDS,
        n_epochs=N_EPOCHS,
        s_mbatch=S_MBATCH,
        lr=LR)
    train_report('Simple convolutional neural network', model, N_ROUNDS,
                 N_EPOCHS, trn_err, tst_err, trn_time, trn_loss, tst_acc)

    # Convolutional neural network with weight sharing

    # Convolutional neural network with auxiliary bias



if __name__ == '__main__':
    main()