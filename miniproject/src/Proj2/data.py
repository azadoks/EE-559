#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data generation for EE-559 miniproject 2."""

__author__ = "Austin Zadoks"

import math

from torch import empty


def generate_data(n: int):
    c = 0.5
    r = 1 / math.sqrt(2 * math.pi)

    input = empty((n * 2, 2)).uniform_(0, 1)
    output = input.sub(c).pow(2).sum(1).sqrt().lt(r).mul(1).unsqueeze(1)

    return (input[:n], output[:n]), (input[n:], output[n:])