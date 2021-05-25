#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import torch
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue

######################################################################

class Net(nn.Module):
    def __init__(self, n_hidden=200):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 10)

    # pylint: disable=arguments-differ
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x


def train_model(model, train_input, train_target, mini_batch_size):
    criterion = nn.MSELoss()
    eta = 1e-1

    for b in range(0, train_input.size(0), mini_batch_size):
        output = model(train_input.narrow(0, b, mini_batch_size))
        loss = criterion(output, train_target.narrow(0, b, mini_batch_size))

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                p -= eta * p.grad


def compute_nb_errors(model, x, y, mini_batch_size):
    nb_errors = 0

    for b in range(0, x.size(0), mini_batch_size):
        output = model(x.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)
        for i in range(mini_batch_size):
            if y[b + i, predicted_classes[i]] <= 0:
                nb_errors += 1

    return nb_errors


if __name__ == '__main__':

    def question_2():
        model = Net()
        mini_batch_size = 100
        for e in range(10):
            train_model(model, train_input, train_target, mini_batch_size)
            nb_train_errors = compute_nb_errors(model, train_input, train_target, mini_batch_size)
            nb_test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)

            print(e, nb_train_errors/train_input.size(0), nb_test_errors/test_input.size(0))


    def question_3():
        pass


    def question_4():
        pass


    train_input, train_target, test_input, test_target = \
        prologue.load_data(one_hot_labels = True, normalize = True, flatten = False)

    question_2()
    question_3()
    question_4()