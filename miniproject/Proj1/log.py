#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for reporting status and results to the user for EE-559 miniproject 1."""

__author__ = "Austin Zadoks"

import torch

try:
    import matplotlib.pyplot as plt
    from math import sqrt, ceil
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import torchsummary
    SUMMARY_AVAILABLE = True
except ImportError:
    SUMMARY_AVAILABLE = False

N_COLS = 100


def _write_block(block: str, title: str):
    if title:
        n_hashes = N_COLS - (len(title) + 2)
        hashes = ''.join(['='] * (n_hashes // 2))
        block = f'{hashes} {title} {hashes}\n' + block + '\n'
    else:
        n_hashes = N_COLS
        hashes = ''.join(['='] * n_hashes)
        block = f'{hashes}\n' + block + '\n'

    return block


# pylint: disable=too-many-arguments
def report_data(trn_inp: torch.Tensor, trn_out: torch.Tensor,
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
    block += ''.join(['-'] * 94) + '\n'
    for data_desc in data:
        block += row_format.format(**data_desc)
    print(_write_block(block, 'DATA'))


def report_model(model):
    print(_write_block(f'{model()}', 'MODEL'))
    if SUMMARY_AVAILABLE:
        torchsummary.summary(model(), (2, 14, 14))


# pylint: disable=too-many-arguments
def report_train(model, n_rounds: int, n_epochs: int,
                 trn_time: torch.Tensor, trn_err: torch.Tensor,
                 tst_err: torch.Tensor, trn_loss_hist: list, tst_acc_hist: list):
    
    block  = ''.join(['-'] * 42) + '\n'
    block += f'Quantity          Value\n'
    block += ''.join(['='] * 42) + '\n'
    block += f'Aux. loss:        {str(model.aux_loss):8s}\n'
    block += f'Share weight:     {str(model.share_weight):8s}\n'
    block += f'No. rounds:       {n_rounds:8d}\n'
    block += f'No. epochs:       {n_epochs:8d}\n'
    block += f'Train error:      {trn_err.mean():8.4f} +- {trn_err.std():8.4f} [%]\n'
    block += f'Test error:       {tst_err.mean():8.4f} +- {tst_err.std():8.4f} [%]\n'
    block += f'Time / round:     {trn_time.mean():8.4f} +- {trn_time.std():8.4f} [s]\n'
    block += f'Total time:       {trn_time.sum():8.4f} [s]\n'

    print(_write_block(block, title='TRAIN'))
