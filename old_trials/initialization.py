# This function initialize the first population of search agents
import numpy as np


def initialization(N, dim, ub, lb):
    if type(ub) is list:
        Boundary_no = len(ub)
    else:
        Boundary_no = 1
    # If the boundaries of all variables are equal and user enter a signle
    # number for both ub and lb
    if Boundary_no == 1:
        X = np.multiply(np.random.rand(N, dim), (ub - lb)) + lb

    # If each variable has a different lb and ub
    if Boundary_no > 1:
        for i in np.arange(1, dim + 1).reshape(-1):
            ub_i = ub[i]
            lb_i = lb[i]
            X[:, i] = np.multiply(np.random.rand(N, 1), (ub_i - lb_i)) + lb_i

    return X
