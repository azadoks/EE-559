import math
import torch

import dlc_practical_prologue as prologue


def sigma(x):
    return x.tanh()


def dsigma(x):
    return x.cosh().pow(-2)


def loss(v, t):
    return (t - v).pow(2).sum()


def dloss(v, t):
    return 2 * (v - t)


def forward_pass(w1, b1, w2, b2, x):
    x0 = x
    s1 = w1.mv(x0) + b1
    x1 = sigma(s1)
    s2 = w2.mv(x1) + b2
    x2 = sigma(s2)
    return x0, s1, x1, s2, x2


# pylint: disable=too-many-arguments
def backward_pass(w1, b1, w2, b2, t, x, s1, x1, s2, x2, dl_dw1, dl_db1, dl_dw2,
                  dl_db2):
    x0 = x
    dl_dx2 = dloss(x2, t)
    dl_ds2 = dsigma(s2) * dl_dx2
    dl_dx1 = w2.t().mv(dl_ds2)
    dl_ds1 = dsigma(s1) * dl_dx1

    dl_dw2.add_(dl_ds2.view(-1, 1).mm(x1.view(1, -1)))
    dl_db2.add_(dl_ds2)
    dl_dw1.add_(dl_ds1.view(-1, 1).mm(x0.view(1, -1)))
    dl_db1.add_(dl_ds1)


if __name__ == '__main__':

    def main():
        train_input, train_target, test_input, test_target = prologue.load_data(
            one_hot_labels=True, normalize=True)

        n_train_input = train_input.size(0)
        n_train_output = train_target.size(1)

        zeta = 0.9
        epsilon = 1e-16
        n_steps = 1000
        eta = 1e-1 / n_train_input
        n_hidden = 50

        print(
            f'\nPARAMETERS\n'
            f'  zeta = {zeta}\n  epsilon = {epsilon}\n  n_steps = {n_steps}\n'
            f'  eta = {eta}\n  n_hidden = {n_hidden}\n')

        train_target.mul_(zeta)
        test_target.mul_(zeta)

        w1 = torch.empty(n_hidden, train_input.size(1)).normal_(0, epsilon)
        b1 = torch.empty(n_hidden).normal_(0, epsilon)
        w2 = torch.empty(n_train_output, n_hidden).normal_(0, epsilon)
        b2 = torch.empty(n_train_output).normal_(0, epsilon)

        dl_dw1 = torch.empty(w1.size())
        dl_db1 = torch.empty(b1.size())
        dl_dw2 = torch.empty(w2.size())
        dl_db2 = torch.empty(b2.size())

        print('RUNNING')
        print('  Step  Acc. Loss  Train Err.  Test Err.')
        for i in range(n_steps):

            dl_dw1.zero_()
            dl_db1.zero_()
            dl_dw2.zero_()
            dl_db2.zero_()

            acc_loss = 0
            n_train_errors = 0
            n_test_errors = 0

            for j in range(train_input.size(0)):

                x0, s1, x1, s2, x2 = forward_pass(w1, b1, w2, b2,
                                                  train_input[j])

                pred_idx = x2.max(dim=0).indices.item()
                if train_target[j, pred_idx] < 0.5:
                    n_train_errors += 1
                acc_loss += loss(x2, train_target[j])

                backward_pass(w1, b1, w2, b2, train_target[j], x0, s1, x1, s2,
                              x2, dl_dw1, dl_db1, dl_dw2, dl_db2)

            w1 = w1 - eta * dl_dw1
            b1 = b1 - eta * dl_db1
            w2 = w2 - eta * dl_dw2
            b2 = b2 - eta * dl_db2

            for j in range(test_input.size(0)):
                _, _, _, _, x2 = forward_pass(w1, b1, w2, b2, test_input[j])

                pred_idx = x2.max(dim=0).indices.item()
                if test_target[j, pred_idx] < 0.5:
                    n_test_errors += 1

            print(f'  {i+1:04d}'
                  f'  {acc_loss:9.3f}'
                  f'  {n_train_errors/train_input.size(0):10.3f}'
                  f'  {n_test_errors/test_input.size(0):9.3f}')

    main()
