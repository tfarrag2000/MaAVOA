import numpy as np
from pymoo.core.repair import Repair


class myRepair(Repair):

    def _do(self, problem, pop, **kwargs):
        # the packing plan for the whole population (each row one individual)
        Z = pop.get("X")
        m=problem.m
        for z in Z:
            for i in range(m):
                z[i + m] = float(np.round(z[i + m]))

        # set the design variables for the population
        pop.set("X", Z)
        return pop