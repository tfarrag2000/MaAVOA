import numpy as np


def random_select(current_vulture_X, Best_vulture1_X, Best_vulture2_X):
    dist1 = np.linalg.norm(current_vulture_X - Best_vulture1_X)
    dist2 = np.linalg.norm(current_vulture_X - Best_vulture2_X)
    if (dist1 <= dist2):
        return Best_vulture1_X
    else:
        return Best_vulture2_X
