import numpy as np

from rouletteWheelSelection import rouletteWheelSelection


def random_select(Best_vulture1_X=None, Best_vulture2_X=None, alpha=None, betha=None):
    probabilities = np.array([alpha, betha])
    if (rouletteWheelSelection(probabilities) == 1):
        random_vulture_X = Best_vulture1_X
    else:
        random_vulture_X = Best_vulture2_X

    return random_vulture_X
