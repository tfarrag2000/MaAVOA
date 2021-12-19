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


def SortPopulation(pop, n):
    # CrownDis=[]
    Ranks = []
    for p in pop:
        # CrownDis.append(p.CrowdingDistance)
        Ranks.append(p.Rank)

    # # Sort Based on Crowding Distance
    # CDSO=np.argsort(CrownDis);
    # CDSO=CDSO[::-1]
    # pop = [x for _,x in sorted(zip(CDSO,pop))]
    # # Sort Based on Rank
    # RSO = np.argsort(Ranks);
    # p2 = [x for _, x in sorted(zip(RSO, pop))]
    # Update Fronts

    MaxRank = np.amax(Ranks)
    F = []
    new_pop = []
    k = 0
    for i in range(0, MaxRank + 1):
        if k == n:
            break
        F.append([])
        for p in pop:
            if p.Rank == i:
                new_pop.append(p)
                F[i].append(len(new_pop) - 1)
                k = k + 1
                if k == n:
                    break

    return new_pop, F
