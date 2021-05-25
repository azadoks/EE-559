#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for tranining neural networks for EE-559 miniproject 1."""

__author__ = "Austin Zadoks"

import torch

import models


def n_errors(model, inp, out, s_mbatch):
    """Calculate number of errors using max selector."""
    model.train(False)
    n_errors = 0

    for b in range(0, inp.size(0), s_mbatch):
        inp_batch = inp.narrow(0, b, s_mbatch)
        out_batch = out.narrow(0, b, s_mbatch)

        pred_out, _, _ = model(inp_batch)
        _, pred_cls = pred_out.max(1)
        n_errors += (out_batch != pred_cls).sum()

    return n_errors


def pct_error(model, inp, out, s_mbatch):
    return n_errors(model, inp, out, s_mbatch) / inp.size(0) * 100


def train_proj1net(
    model: models.Proj1Net,
    optimizer,
    criterion,
    trn_inp: torch.Tensor,
    trn_out: torch.Tensor,
    trn_cls: torch.Tensor,
    tst_inp: torch.Tensor,
    tst_out: torch.Tensor,
    tst_cls: torch.Tensor,
    n_epochs,
    s_mbatch,
    track=False):
    """Train a LeNet-style convolutional neural network to learn if the digit in the
    first MNIST image in a pair is lesser or equal to the digit in the second
    image."""
    model.train(True)
    # Initialize outputs
    # pylint: disable=no-member
    if model.aux_loss:
        trn_loss = torch.empty((n_epochs,3), dtype=torch.float)
        tst_acc = torch.empty((n_epochs,3), dtype=torch.float)
    else:
        trn_loss = torch.empty((n_epochs,), dtype=torch.float)
        tst_acc = torch.empty((n_epochs,), dtype=torch.float)
    # pylint: enable=no-member
    # Train for some epochs
    for e in range(n_epochs):
        # Iterate over minibatches
        for b in range(0, trn_inp.size(0), s_mbatch):
            # Get minibatch data
            trn_inp_batch = trn_inp.narrow(0, b, s_mbatch)
            trn_out_batch = trn_out.narrow(0, b, s_mbatch)
            trn_cls_batch = trn_cls.narrow(0, b, s_mbatch)
            # Run forward pass
            # pred_out1 and pred_out2 are None if aux_loss=False
            pred_batch, pred_batch1, pred_batch2 = model(trn_inp_batch)
            # Calculate loss
            loss = criterion(pred_batch, trn_out_batch)
            if model.aux_loss:
                loss += criterion(pred_batch1, trn_cls_batch[:,0])
                loss += criterion(pred_batch2, trn_cls_batch[:,1])
            # Run backward pass and step the optimizer
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # After each batch, calculate train and test loss and accuracy
        # TODO: calculate accuracy
        # Train loss and accuracy
        if track:
            pred_trn, pred_trn1, pred_trn2 = model(trn_inp)
            if model.aux_loss:
                trn_loss[e,0] = criterion(pred_trn, trn_out)
                trn_loss[e,1] = criterion(pred_trn1, trn_cls[:,0])  # Column 0 is 1st digit classes
                trn_loss[e,2] = criterion(pred_trn2, trn_cls[:,1])  # Column 1 is 2nd digit classes
            else:
                trn_loss[e] = criterion(pred_trn, trn_out)
            model.train(False)
            # pred_tst, pred_tst1, pred_tst2 = model(tst_inp)
            if model.aux_loss:
                tst_acc[e,0] = 100 - pct_error(model, tst_inp, tst_out, s_mbatch)
                tst_acc[e,1] = 100 - pct_error(model, tst_inp, tst_cls[:,0], s_mbatch)
                tst_acc[e,2] = 100 - pct_error(model, tst_inp, tst_cls[:,1], s_mbatch)
            else:
                tst_acc[e] = 100 - pct_error(model, tst_inp, trn_out, s_mbatch)
            model.train(True)

    return trn_loss, tst_acc
