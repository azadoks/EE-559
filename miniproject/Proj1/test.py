#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main executable for EE-559 miniproject 1."""

__author__ = "Austin Zadoks"

import time
import warnings

import torch
from torch import nn, optim

import models
import dlc_practical_prologue as prologue
import log
import train
import plot

warnings.filterwarnings('ignore')
# torch.set_num_threads(1)  # Ensure a fair environment for timing  pylint: disable=no-member


def main():
    N_PAIRS = 1000  # Number of pairs, specified by miniproject prompt
    N_ROUNDS = 10  # Number of rounds to train each model
    N_EPOCHS = 25  # Number of epochs
    S_MBATCH = 100  # Minibatch size
    MODEL = models.Proj1Net
    OPTIMIZER = optim.Adam
    OPTIMIZER_PARAMS = {'lr': 1e-3}  # Optimizer parameters
    CRITERION = nn.CrossEntropyLoss
    TRACK = False  # Track loss and accuracy
    
    seeds = torch.arange(2021, 2022 + N_ROUNDS, 1)

    for share_weight, aux_loss in [(False, False), (True, False),
                                   (False, True), (True, True)]:

        trn_loss_hist = torch.empty((N_EPOCHS, N_ROUNDS), dtype=torch.float)
        tst_acc_hist = torch.empty((N_EPOCHS, N_ROUNDS), dtype=torch.float)

        trn_err = torch.empty((N_ROUNDS,), dtype=torch.float)  # pylint: disable=no-member
        tst_err = torch.empty((N_ROUNDS,), dtype=torch.float)  # pylint: disable=no-member
        trn_time = torch.empty((N_ROUNDS,), dtype=torch.float)  # pylint: disable=no-member
        
        for r in range(N_ROUNDS):
            torch.manual_seed(seeds[r])

            trn_inp, trn_out, trn_cls, tst_inp, tst_out, tst_cls = prologue.generate_pair_sets(
                N_PAIRS)
            model = MODEL(share_weight=share_weight, aux_loss=aux_loss)
            optimizer = OPTIMIZER(model.parameters(), **OPTIMIZER_PARAMS)
            criterion = CRITERION()
            
            t0 = time.perf_counter()
            model.train(True)
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

            model.train(False)
            with torch.no_grad():
                tmp_trn_err = train.calculate_error_batch(model, trn_inp, trn_out, trn_cls, S_MBATCH) * 100
                tmp_tst_err = train.calculate_error_batch(model, tst_inp, tst_out, tst_cls, S_MBATCH) * 100
                if aux_loss:
                    trn_err[r] = tmp_trn_err[0]
                    tst_err[r] = tmp_tst_err[0]
                else:
                    trn_err[r] = tmp_trn_err
                    tst_err[r] = tmp_tst_err
            
            if TRACK:
                if aux_loss:
                    trn_loss_hist[:,r] = trn_loss[:,0]
                    tst_acc_hist[:,r] = tst_acc[:,0]
                else:
                    trn_loss_hist[:,r] = trn_loss
                    tst_acc_hist[:,r] = tst_acc
                plot.plot_loss_acc(trn_loss_hist, tst_acc_hist,
                                   title=f'Weight sharing = {share_weight}, Aux. loss = {aux_loss}')

        log.report_train(model, N_ROUNDS, N_EPOCHS, trn_time, trn_err,
                         tst_err, trn_loss_hist, tst_acc_hist)


if __name__ == '__main__':
    main()
