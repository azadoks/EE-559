#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data loading functions for EE-559 miniproject 1."""

__author__ = "Austin Zadoks"

import typing as ty

import torch
import torch.nn.functional as F
from torch.utils import data

import dlc_practical_prologue as prologue


class ImagePairDataset(data.Dataset):
    """
    A custom Dataset class for storing MNIST image pair data.
    
    :param input: MNIST image pairs
    :param target: binary target
    :param classes: classes of images in the image pairs
    """
    def __init__(self, input: torch.Tensor, target: torch.Tensor, classes: torch.Tensor):
        self.input = input
        self.target = target  # image 1 digit <= image 2 digit ?
        self.class1 = classes[:,0]  # image 1 class
        self.class2 = classes[:,1]  # image 2 class

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx: int) -> ty.Tuple[torch.Tensor, ty.Tuple[torch.Tensor]]:
        return self.input[idx], (self.target[idx], self.class1[idx], self.class2[idx])


def load_data(n_pairs: int, batch_size: int) -> ty.Tuple[data.DataLoader]:
    """
    Load MNIT image pair data.
    
    :param n_pairs: number of image pairs
    :param batch_size: DataLoader batch size
    """
    # pair_sets = (train_input, train_target, train_classes, test_input, test_target, test_classes)
    pair_sets = prologue.generate_pair_sets(n_pairs)

    train_data = ImagePairDataset(pair_sets[0], pair_sets[1], pair_sets[2])
    test_data = ImagePairDataset(pair_sets[3], pair_sets[4], pair_sets[5])

    train_loader = data.DataLoader(train_data, batch_size=batch_size)
    test_loader = data.DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader
