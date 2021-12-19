import numpy as np


def exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound):
    if np.random.rand() < p1:
        current_vulture_X = random_vulture_X - (
            np.abs((2 * np.random.rand()) * random_vulture_X - current_vulture_X)) * F
    else:
        current_vulture_X = (
                random_vulture_X - (F) + np.random.rand() * (
                (upper_bound - lower_bound) * np.random.rand() + lower_bound))

    return current_vulture_X
