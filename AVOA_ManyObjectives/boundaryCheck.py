import numpy as np


def boundaryCheck(X, lb, ub):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if (X[i, j] > ub) or (X[i, j] < lb):
                X[i, j] = np.multiply(np.random.rand(1, 1), (ub - lb)) + lb

    return X
