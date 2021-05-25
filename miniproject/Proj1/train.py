#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for tranining neural networks for EE-559 miniproject 1."""

__author__ = "Austin Zadoks"

import torch

import models


def calculate_n_errors(pred_out, out):
    return (pred_out.argmax(1) != out).sum()


def calculate_error(pred_out, out):
    n_errors = calculate_n_errors(pred_out, out)
    return n_errors / out.size(0)


def calculate_accuracy(pred_out, out):
    return 1 - calculate_error(pred_out, out)


def calculate_error_batch(model, inp, out, cls, s_mbatch):
    if model.aux_loss:
        err = torch.zeros((3,), dtype=torch.float)  # pylint: disable=no-member
    else:
        err = 0

    n_batch = 0
    for b in range(0, inp.size(0), s_mbatch):
        n_batch += 1

        inp_batch = inp.narrow(0, b, s_mbatch)
        out_batch = out.narrow(0, b, s_mbatch)
        cls_batch = cls.narrow(0, b, s_mbatch)

        if model.aux_loss:
            pred_batch, pred_aux_batch = model(inp_batch)
            err[0] += calculate_error(pred_batch, out_batch)
            err[1] += calculate_error(pred_aux_batch[0], cls_batch[:,0])
            err[2] += calculate_error(pred_aux_batch[1], cls_batch[:,1])
        else:
            pred_batch, _ = model(inp_batch)
            err += calculate_error(pred_batch, out_batch)

    err = err / n_batch
    return err

def calculate_loss_accuracy_batch(model, criterion, trn_inp, trn_out, trn_cls, tst_inp, tst_out,
                                  tst_cls, s_mbatch):
    if model.aux_loss:
        trn_loss = torch.zeros((3, ), dtype=torch.float)  # pylint: disable=no-member
        tst_acc = torch.zeros((3, ), dtype=torch.float)  # pylint: disable=no-member
    else:
        trn_loss = 0
        tst_acc = 0

    for b in range(0, trn_inp.size(0), s_mbatch):
        trn_inp_batch = trn_inp.narrow(0, b, s_mbatch)
        trn_out_batch = trn_out.narrow(0, b, s_mbatch)
        trn_cls_batch = trn_cls.narrow(0, b, s_mbatch)
        tst_inp_batch = tst_inp.narrow(0, b, s_mbatch)
        tst_out_batch = tst_out.narrow(0, b, s_mbatch)
        tst_cls_batch = tst_cls.narrow(0, b, s_mbatch)

        pred_trn_batch, pred_trn_aux_batch = model(trn_inp_batch)
        pred_tst_batch, pred_tst_aux_batch = model(tst_inp_batch)

        if model.aux_loss:
            trn_loss[0] += criterion(pred_trn_batch, trn_out_batch)
            trn_loss[1] += criterion(pred_trn_aux_batch[0], trn_cls_batch[:,0])
            trn_loss[2] += criterion(pred_trn_aux_batch[1], trn_cls_batch[:,1])
            tst_acc[0] += calculate_accuracy(pred_tst_batch, tst_out_batch)
            tst_acc[1] += calculate_accuracy(pred_tst_aux_batch[0], tst_cls_batch[:, 0])
            tst_acc[2] += calculate_accuracy(pred_tst_aux_batch[1], tst_cls_batch[:, 1])
        else:
            trn_loss += criterion(pred_trn_batch, trn_out_batch)
            tst_acc += calculate_accuracy(pred_tst_batch, tst_out_batch)

    return trn_loss, tst_acc


def train_proj1net(model: models.Proj1Net,
                   optimizer,
                   criterion,
                   trn_inp: torch.Tensor,
                   trn_out: torch.Tensor,
                   trn_cls: torch.Tensor,
                   tst_inp: torch.Tensor,
                   tst_out: torch.Tensor,
                   tst_cls: torch.Tensor,
                   n_epochs: int,
                   s_mbatch: int,
                   track: bool=False):
    """Train a LeNet-style convolutional neural network to learn if the digit in the
    first MNIST image in a pair is lesser or equal to the digit in the second
    image."""
    model.train(True)

    if track:
        # pylint: disable=no-member
        if model.aux_loss:
            trn_loss = torch.empty((n_epochs, 3), dtype=torch.float)
            tst_acc = torch.empty((n_epochs, 3), dtype=torch.float)
        else:
            trn_loss = torch.empty((n_epochs, ), dtype=torch.float)
            tst_acc = torch.empty((n_epochs, ), dtype=torch.float)
        # pylint: enable=no-member
    else:
        trn_loss, tst_acc = None, None

    for e in range(n_epochs):
        for b in range(0, trn_inp.size(0), s_mbatch):
            trn_inp_batch = trn_inp.narrow(0, b, s_mbatch)
            trn_out_batch = trn_out.narrow(0, b, s_mbatch)
            trn_cls_batch = trn_cls.narrow(0, b, s_mbatch)

            pred_batch, pred_batch_aux = model(trn_inp_batch)
            # Calculate loss
            loss = criterion(pred_batch, trn_out_batch)
            if model.aux_loss:
                loss += criterion(pred_batch_aux[0], trn_cls_batch[:, 0])
                loss += criterion(pred_batch_aux[1], trn_cls_batch[:, 1])
            # Run backward pass and step the optimizer
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # For each epoch, calculate train and test loss and accuracy
        if track:
            model.train(False)
            with torch.nograd():  # pylint: disable=no-member
                tmp_loss, tmp_acc = calculate_loss_accuracy_batch(
                    model, criterion, trn_inp, trn_out, trn_cls, tst_inp,
                    tst_out, tst_cls, s_mbatch)
                if model.aux_loss:
                    trn_loss[e, :] = tmp_loss
                    tst_acc[e, :] = tmp_acc
                else:
                    trn_loss[e] = tmp_loss
                    tst_acc[e] = tmp_acc
            model.train(True)

    return trn_loss, tst_acc
