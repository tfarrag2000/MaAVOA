import math

import numpy as np
import scipy.special


def levyFlight(d):
    beta = 3 / 2
    sigma = (scipy.special.gamma(1 + beta) * np.sin(math.pi * beta / 2) / (
            scipy.special.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(1, d) * sigma
    v = np.random.randn(1, d)
    step = u / np.abs(v) ** (1 / beta)
    o = step
    return o
