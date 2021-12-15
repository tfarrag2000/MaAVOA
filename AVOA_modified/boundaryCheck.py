import numpy as np


def boundaryCheck(X=None, lb=None, ub=None):
    for i in range(X.shape[0]):
        FU = X[i, :] > ub
        FL = X[i, :] < lb
        X[i, :] = (np.multiply(X[i, :], np.invert(FU + FL))) + np.multiply(ub, FU) + np.multiply(lb, FL)
    return X
