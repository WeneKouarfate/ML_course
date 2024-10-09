# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    gram = tx.T @ tx
    w_star = np.linalg.solve(gram, tx.T @ y)

    e = y - tx.dot(w_star)
    mse = (e @ e.T) / (2 * len(e))

    return mse, w_star
