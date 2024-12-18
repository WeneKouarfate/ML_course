# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    stack = np.column_stack([x for _ in range(degree+1)])
    return np.power(stack, np.arange(degree+1))
