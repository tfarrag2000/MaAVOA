#
# Copyright (c) 2015, Yarpiz (www.yarpiz.com)
# All rights reserved. Please read the "license.txt" for license terms.
#
# Project Code: YPEA120
# Project Title: Non-dominated Sorting Genetic Algorithm II (NSGA-II)
# Publisher: Yarpiz (www.yarpiz.com)
#
# Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
#
# Contact Info: sm.kalami@gmail.com, info@yarpiz.com
#

import numpy as np


def Dominates(x, y, AccordingTo):
    # if isinstance(x, empty_individual):
    #     x = x.Cost
    # if isinstance(y, empty_individual):
    #     y = y.Cost

    if AccordingTo == 0:
        x = x.Cost
        y = y.Cost
    elif AccordingTo == 1:
        x = x.Costobj
        y = y.Costobj

    b1 = np.all(x <= y)
    b2 = np.any(x < y)
    b = (b1 and b2)
    return b
