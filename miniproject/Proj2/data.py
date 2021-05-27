#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data generation for EE-559 miniproject 2."""

__author__ = "Austin Zadoks"

import math

from torch import empty


def generate_data(n: int):
    c = 0.5  # center of the circle at [c, c]
    r = 1 / math.sqrt(2 * math.pi)  # radius of the circle 1 / sqrt(2pi)

    data = empty((n * 2, 2)).uniform_(0, 1)  # uniformly sample points in the unit 1st quadrant
    # norm(points - center) < radius -> float; add 1st dimension to make a matrix
    target = data.sub(c).pow(2).sum(1).sqrt().lt(r).float().unsqueeze(1)
    # (train_data, train_target), (test_data, test_target)
    return (data[:n], target[:n]), (data[n:], target[n:])