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

from old_trials.NonDominatedSorting.Dominates import Dominates


def NonDominatedSorting(pop, n, AccordingTo):  # AccordingTo=0 by Conv and Diver ,  AccordingTo=1 by objective matrix
    nPop = np.asarray(pop).size
    for i in range(nPop):
        pop[i].DominationSet.clear()
        pop[i].DominatedCount = 0

    F = []
    F.append([])
    for i in range(nPop):
        for j in range(i + 1, nPop):
            p = pop[i]
            q = pop[j]

            if Dominates(p, q, AccordingTo):
                p.DominationSet.append(j)
                q.DominatedCount = q.DominatedCount + 1
            if Dominates(q, p, AccordingTo):
                q.DominationSet.append(i)
                p.DominatedCount = p.DominatedCount + 1

            pop[i] = p
            pop[j] = q
        if pop[i].DominatedCount == 0:
            F[0].append(i)
            pop[i].Rank = 0

    k = 0
    while True:
        Q = []
        for i in F[k]:
            p = pop[i]
            for j in p.DominationSet:
                q = pop[j]
                q.DominatedCount = q.DominatedCount - 1
                if q.DominatedCount == 0:
                    Q.append(j)
                    q.Rank = k + 1
                pop[j] = q
        if len(Q) == 0:
            break
        F.append(Q)
        k = k + 1
    return pop, F
