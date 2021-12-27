import math
import random

import numpy as np

from pymoo.core.mutation import Mutation
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem

from AVOA_ManyObjectives.boundaryCheck import boundaryCheck
from AVOA_ManyObjectives.exploitation import exploitation
from AVOA_ManyObjectives.exploration import exploration
from AVOA_ManyObjectives.NonDominatedSorting.EvaluatePopulation import evaluatePopulation
from AVOA_ManyObjectives.random_select import random_select


class AfricanMutation(Mutation):
    def __init__(self, eta, prob=None):
        super().__init__()
        self.eta = float(eta)

        if prob is not None:
            self.probability = float(prob)
        else:
            self.probability = None

    def _do(self, problem, X, **kwargs):

        X = X.astype(float)
        # Y = np.full(X.shape, np.inf)

        if self.probability is None:
            self.probability = 1.0 / problem.n_var
        do_mutation = np.random.random(X.shape[0]) < self.probability

        _, _, pop, F_Rank = evaluatePopulation(X, X.shape[0], problem.evaluate,AccordingTo=1)
        current_iter=1
        max_iter=1
        upper_bound=problem.xu[0]
        lower_bound=problem.xl[0]
        ##  Controlling parameter
        p1 = 0.6
        p2 = 0.4
        p3 = 0.6
        alpha = 0.8
        betha = 0.2
        gamma = 2.5

        Best_V1_id = random.choice(F_Rank[0])
        if len(F_Rank) == 1:
            Best_V2_id = random.choice(F_Rank[0])
        else:
            Best_V2_id = random.choice(F_Rank[1])
        Best_V1_individual = pop[Best_V1_id]
        Best_V2_individual = pop[Best_V2_id]
        Best_V1_X = Best_V1_individual.Position.reshape((1, X.shape[1]))
        Best_V2_X = Best_V2_individual.Position.reshape((1, X.shape[1]))

        ################### Africian exploration & exploitation ###################
        a = np.random.uniform(- 2, 2, (1, 1)) * ((np.sin((math.pi / 2) * (current_iter / max_iter)) ** gamma) + np.cos(
            (math.pi / 2) * (current_iter / max_iter)) - 1)
        P1 = (2 * np.random.rand() + 1) * (1 - (current_iter / max_iter)) + a
        # Update the location
        for i in range(X.shape[0]):
            if do_mutation[i]==True:
                current_V_X = X[i, :]
                F = P1 * (2 * np.random.rand() - 1)
                random_V_X = random_select(current_V_X, Best_V1_X, Best_V2_X)
                if np.abs(F) >= 1:
                    current_V_X = exploration(current_V_X, random_V_X, F, p1, upper_bound, lower_bound)
                else:
                    if np.abs(F) < 1:
                        current_V_X = exploitation(current_V_X, Best_V1_X, Best_V2_X,
                                                   random_V_X, F, p2, p3, X.shape[1], upper_bound,
                                                   lower_bound)
                X[i, :] = current_V_X
        # in case out of bounds repair (very unlikely)
        # Y=boundaryCheck(X,upper_bound, lower_bound)
        Y = set_to_bounds_if_outside_by_problem(problem, X)

        return Y


class AM(AfricanMutation):
    pass