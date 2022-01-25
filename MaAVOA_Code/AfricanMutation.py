import math
import random

import numpy as np
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.mutation import Mutation
from pymoo.core.population import pop_from_array_or_individual
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem

from MaAVOA.exploitation import exploitation
from MaAVOA.exploration import exploration


class AfricanMutation(Mutation):
    def __init__(self, eta, ref_dirs, prob=None):
        super().__init__()
        self.eta = float(eta)
        self.ARC = None
        if prob is not None:
            self.probability = float(prob)
        else:
            self.probability = None
        self.ref_dirs = ref_dirs

    def _do(self, problem, X, **kwargs):

        X = X.astype(float)
        # Y = np.full(X.shape, np.inf)

        if self.probability is None:
            self.probability = 1.0 / problem.n_var
        do_mutation = np.random.random(X.shape[0]) < self.probability

        F = problem.evaluate(X)
        pop_unsorted = pop_from_array_or_individual(X)
        pop_unsorted.set('F', F)

        survival = ReferenceDirectionSurvival(self.ref_dirs)
        pop = survival.do(problem, pop_unsorted, n_survive=X.shape[0])

        # FP_iter = []
        V2_Leaders = np.full((F.shape[1],), None)
        Top_F = np.full((F.shape[1],), np.inf)

        for p in pop:
            # rank = p.data['rank']
            # if rank == 0:
            #     FP_iter.append(p)
            for i in range(len(Top_F)):
                if p.F[i] < Top_F[i]:
                    Top_F[i] = p.F[i]
                    V2_Leaders[i] = p

        FP_iter1 = survival.opt

        self.ARC = np.append(self.ARC, FP_iter1)

        Best_V1_X = self.__selectV1(self.ARC.tolist()).X
        Best_V2_X = self.__selectV2(V2_Leaders).X
        variables_no = X.shape[1]

        current_iter = 1
        max_iter = 1

        X = np.array([p.X for p in pop])
        upper_bound = problem.xu[0]
        lower_bound = problem.xl[0]
        ##  Controlling parameter
        p1 = 0.6
        p2 = 0.4
        p3 = 0.6
        alpha = 0.8
        betha = 0.2
        gamma = 2.5
        ################### Africian exploration & exploitation ###################
        a = np.random.uniform(- 2, 2, (1, 1)) * ((np.sin((math.pi / 2) * (current_iter / max_iter)) ** gamma) + np.cos(
            (math.pi / 2) * (current_iter / max_iter)) - 1)
        P1 = (2 * np.random.rand() + 1) * (1 - (current_iter / max_iter)) + a
        # Update the location
        for i in range(X.shape[0]):
            if do_mutation[i] == True:
                current_V_X = X[i, :]
                F = P1 * (2 * np.random.rand() - 1)
                random_V_X = random.choice([Best_V1_X, Best_V2_X])
                if np.abs(F) >= 1:
                    current_V_X = exploration(current_V_X, random_V_X, F, p1, upper_bound, lower_bound)
                else:
                    if np.abs(F) < 1:
                        current_V_X = exploitation(current_V_X, Best_V1_X, Best_V2_X,
                                                   random_V_X, F, p2, p3, variables_no, upper_bound,
                                                   lower_bound)
                X[i, :] = current_V_X
        # in case out of bounds repair (very unlikely)
        # Y=boundaryCheck(X,upper_bound, lower_bound)
        Y = set_to_bounds_if_outside_by_problem(problem, X)

        return Y

    def __selectV1(self, V1_Leaders):
        V1_Leaders = list(filter(None, set(V1_Leaders)))
        if (len(V1_Leaders) < 2):
            return V1_Leaders[0]
        V1_1, V1_2 = random.sample(V1_Leaders, k=2)

        if V1_1.data['niche'] < V1_2.data['niche']:
            return V1_1
        else:
            return V1_2

    def __selectV2(self, V2_Leaders):

        return random.choice(V2_Leaders)


class AM(AfricanMutation):
    pass
