import numpy as np


def boundaryCheck(X, lb, ub):
    X_new = np.clip(X, lb, ub)
    return X_new
