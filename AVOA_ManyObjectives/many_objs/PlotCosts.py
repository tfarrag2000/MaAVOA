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

import matplotlib.pyplot as plt
import numpy as np


def PlotCosts(F1 ):
    y1 = np.zeros((np.asarray(F1).size, 1))
    y2 = np.zeros((np.asarray(F1).size, 1))

    for i in range(np.asarray(F1).size):
        y1[i] = F1[i].Cost[0]
        y2[i] = F1[i].Cost[1]

    plt.plot(y1, y2, 'r*', markersize=8)
    plt.xlabel('$\mathregular{1^{st}}$ Objective')
    plt.ylabel('$\mathregular{2^{nd}}$ Objective')
    plt.title('Non-dominated Solutions ($\mathregular{F_{1}}$)')
    plt.grid('on')
    plt.show()
    return
