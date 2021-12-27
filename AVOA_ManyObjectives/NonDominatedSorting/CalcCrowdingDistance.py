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

import math

import numpy as np


def CalcCrowdingDistance(pop, F):
    nF = np.asarray(F, dtype=object).size
    for k in range(nF):
        Costs = []
        for i in F[k]:
            Costs.extend(list(pop[i].Cost))
        Costs = np.array(Costs).reshape((1, -1))
        nObj = 1
        n = Costs.shape[1]
        d = np.zeros((n, nObj))
        # d =np.full((n, nObj), math.inf)
        for j in range(nObj):
            so = list(np.argsort(Costs).flatten())
            cj = sorted(Costs.flatten())
            d[so[0], j] = math.inf
            for i in range(1, n - 1):
                d[so[i], j] = np.abs(cj[i + 1] - cj[i - 1]) / np.abs(cj[0] - cj[-1])
            d[so[-1], j] = math.inf

        for i in range(len(F[k])):
            pop[F[k][i]].CrowdingDistance = sum(d[i, :])

    return pop
