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

from AVOA_ManyObjectives.many_objs.empty_individual import empty_individual


def Dominates(x , y ):
    if isinstance(x, empty_individual):
        x = x.Cost
    if isinstance(y, empty_individual):
        y = y.Cost

    b1 = np.all(x <= y)
    b2 = np.any(x < y)
    b = (b1 and b2)
    return b
