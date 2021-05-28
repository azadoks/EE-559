#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data generation for EE-559 miniproject 2."""

__author__ = "Austin Zadoks"

import math

from torch import empty  # pylint: disable=no-name-in-module


def generate_data(n: int) -> tuple:
    """
    Generate 2D coordinates in the first unit quadrant and binary values for whether
    they lie in a circle of radius 1 / sqrt(2pi) centered at [0.5, 0.5].

    :param n: number of points in each of the training and test data sets
    :returns: (train_input, train_target), (test_input, test_target)
    """
    c = 0.5  # center of the circle at [c, c]
    r = 1 / math.sqrt(2 * math.pi)  # radius of the circle 1 / sqrt(2pi)

    data = empty((n * 2, 2)).uniform_(0, 1)  # uniformly sample points in the unit 1st quadrant
    # norm(points - center) < radius -> float; add 1st dimension to make a matrix
    target = data.sub(c).pow(2).sum(1).sqrt().lt(r).float().unsqueeze(1)
    # (train_input, train_target), (test_input, test_target)
    return (data[:n], target[:n]), (data[n:], target[n:])