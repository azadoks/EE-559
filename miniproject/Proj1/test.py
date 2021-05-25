#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main executable for EE-559 miniproject 1."""

__author__ = "Austin Zadoks"

import time
import warnings

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import models
import dlc_practical_prologue as prologue
import log
import train
import plot

warnings.filterwarnings('ignore')
# torch.set_num_threads(1)  # Ensure a fair environment for timing  pylint: disable=no-member


def main():
    # TODO: ensure random seed for each round
    N_PAIRS = 1000  # Number of pairs, specified by miniproject prompt
    N_ROUNDS = 25  # Number of rounds to train each model
    N_EPOCHS = 25  # Number of epochs
    S_MBATCH = 100  # Minibatch size
    MODEL = models.Proj1Net
    OPTIMIZER = optim.Adam
    OPTIMIZER_PARAMS = {'lr': 1e-2}  # Optimizer parameters
    CRITERION = nn.CrossEntropyLoss
    TRACK = True  # Track loss and accuracy

    # Load data
    # trn_inp, trn_out, trn_cls, tst_inp, tst_out, tst_cls = prologue.generate_pair_sets(N_PAIRS)
    # Print data report (should match table in assigment document pg. 2)
    # data_report = report.report_data(trn_inp, trn_out, trn_cls, tst_inp, tst_out, tst_cls)
    # print(data_report)

    # report.report_model(MODEL)

    for share_weight, aux_loss in [(False, False), (True, False),
                                   (False, True), (True, True)]:
        # title = f'share_weight={share_weight}-aux_loss={aux_loss}'
        # title = ''
        
        # Lists for plotting data
        # trn_loss_hist, tst_loss_hist, trn_acc_hist, tst_acc_hist = [], [], [], []
        trn_loss_hist = torch.empty((N_EPOCHS, N_ROUNDS), dtype=torch.float)
        # tst_loss_hist = torch.empty((N_EPOCHS, N_ROUNDS), dtype=torch.float)
        # trn_acc_hist = torch.empty((N_EPOCHS, N_ROUNDS), dtype=torch.float)
        tst_acc_hist = torch.empty((N_EPOCHS, N_ROUNDS), dtype=torch.float)
        # Tensors for error and time data
        trn_err = torch.empty((N_ROUNDS,), dtype=torch.float)  # pylint: disable=no-member
        tst_err = torch.empty((N_ROUNDS,), dtype=torch.float)  # pylint: disable=no-member
        trn_time = torch.empty((N_ROUNDS,), dtype=torch.float)  # pylint: disable=no-member
        
        # Run some rounds
        for r in range(N_ROUNDS):
            trn_inp, trn_out, trn_cls, tst_inp, tst_out, tst_cls = prologue.generate_pair_sets(
                N_PAIRS)
            model = MODEL(share_weight=share_weight,
                          aux_loss=aux_loss)
            optimizer = OPTIMIZER(model.parameters(), **OPTIMIZER_PARAMS)
            criterion = CRITERION()
            
            t0 = time.perf_counter()
            trn_loss, tst_acc = train.train_proj1net(
                model,
                optimizer,
                criterion,
                trn_inp,
                trn_out,
                trn_cls,
                tst_inp,
                tst_out,
                tst_cls,
                N_EPOCHS,
                S_MBATCH,
                track=TRACK)
            t1 = time.perf_counter()
            
            trn_time[r] = t1 - t0
            trn_err[r] = train.pct_error(model, trn_inp, trn_out, S_MBATCH)
            tst_err[r] = train.pct_error(model, tst_inp, tst_out, S_MBATCH)
            
            if aux_loss:
                trn_loss_hist[:,r] = trn_loss[:,0]
                tst_acc_hist[:,r] = tst_acc[:,0]
            else:
                trn_loss_hist[:,r] = trn_loss
                tst_acc_hist[:,r] = tst_acc

            plot.plot_loss_acc(trn_loss_hist, tst_acc_hist,
                               title=f'Weight sharing = {share_weight}, Aux. loss = {aux_loss}')

        # train_report = report.report_train(title, model, N_ROUNDS, N_EPOCHS, trn_time, trn_err,
        #                                    tst_err, trn_loss_hist, tst_loss_hist, trn_acc_hist,
        #                                    tst_acc_hist)
        # print(train_report)

    return (trn_inp, trn_out, trn_cls, tst_inp, tst_out, tst_cls)


if __name__ == '__main__':
    (trn_inp, trn_out, trn_cls, tst_inp, tst_out, tst_cls) = main()
