def boundaryCheck(X, lb, ub):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] > ub:
                X[i, j] = ub
            elif X[i, j] < lb:
                X[i, j] = lb

    return X
