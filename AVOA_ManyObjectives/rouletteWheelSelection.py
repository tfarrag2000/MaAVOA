import numpy as np


def rouletteWheelSelection(x):
    index = np.nonzero(np.random.rand() <= np.cumsum(x))
    index1 = index[0][0]
    return index1
