import math
import random
from copy import deepcopy

import numpy as np
from pymoo.factory import get_performance_indicator

from AVOA_ManyObjectives.boundaryCheck import boundaryCheck
from AVOA_ManyObjectives.many_objs.EvaluatePopulation import evaluatePopulation
from AVOA_ManyObjectives.many_objs.benchmark import pareto_front
from exploitation import exploitation
from exploration import exploration
from initialization import initialization
from random_select import random_select


def AVOA(pop_size=None, max_iter=None, lower_bound=None, upper_bound=None, variables_no=None):
    # initialize Best_vulture1, Best_vulture2
    # Best_vulture1_X = np.zeros((1, variables_no))
    # print("Best_vulture1_X = ",Best_vulture1_X)
    # Best_vulture1_F = math.inf
    # print("Best_vulture1_F = ",Best_vulture1_F)
    # Best_vulture2_X = np.zeros((1, variables_no))
    # Best_vulture2_F = math.inf
    # Initialize the first random population of vultures
    X = initialization(pop_size, variables_no, upper_bound, lower_bound)
    X_init = deepcopy(X)
    # print("X = ",X)
    ##  Controlling parameter
    p1 = 0.6
    p2 = 0.4
    p3 = 0.6
    alpha = 0.8
    betha = 0.2
    gamma = 2.5
    ##Main loop
    current_iter = 0
    convergence_curve = []
    while current_iter < max_iter:
         # for i in range(X.shape[0]):
        #     # Calculate the fitness of the population
        #     current_vulture_X = X[i, :]
        #     # print("current_vulture_X= ", current_vulture_X)
        #     current_vulture_F = fobj(current_vulture_X)
        #     # print("current_vulture_F= ", current_vulture_F)
        #     # Update the first best two vultures if needed
        #     if current_vulture_F < Best_vulture1_F:
        #         Best_vulture1_F = current_vulture_F
        #         Best_vulture1_X = current_vulture_X
        #     if (current_vulture_F > Best_vulture1_F) and (current_vulture_F < Best_vulture2_F):
        #         Best_vulture2_F = current_vulture_F
        #         Best_vulture2_X = current_vulture_X

        pop, F_Rank = evaluatePopulation(X)
        Best_vulture1_id = random.choice(F_Rank[0])
        Best_vulture2_id = random.choice(F_Rank[1])
        Best_vulture1_individual = pop[Best_vulture1_id]
        Best_vulture2_individual = pop[Best_vulture2_id]
        Best_vulture1_X = Best_vulture1_individual.Position.reshape((1, variables_no))
        Best_vulture2_X = Best_vulture2_individual.Position.reshape((1, variables_no))

        a = np.random.uniform(- 2, 2, (1, 1)) * ((np.sin((math.pi / 2) * (current_iter / max_iter)) ** gamma) + np.cos(
            (math.pi / 2) * (current_iter / max_iter)) - 1)
        P1 = (2 * np.random.rand() + 1) * (1 - (current_iter / max_iter)) + a
        # Update the location
        for i in range(X.shape[0]):
            current_vulture_X = X[i, :]
            F = P1 * (2 * np.random.rand() - 1)
            random_vulture_X = random_select(Best_vulture1_X, Best_vulture2_X, alpha, betha)
            if np.abs(F) >= 1:
                current_vulture_X = exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound)
            else:
                if np.abs(F) < 1:
                    current_vulture_X = exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X,
                                                     random_vulture_X, F, p2, p3, variables_no, upper_bound,
                                                     lower_bound)
            X[i, :] = current_vulture_X
        convergence_curve.append(Best_vulture1_individual.Cost[1])
        current_iter = current_iter + 1

        X = boundaryCheck(X, lower_bound, upper_bound)
        print('In Iteration %d, best estimation of Conversion and Diversion is %4.2f , %4.2f \n ' % (
            current_iter, Best_vulture1_individual.Cost[0], Best_vulture1_individual.Cost[1]))

    pop, F = evaluatePopulation(X)
    X_list =[ x.Position for x in pop  ]

    ############ IGD ############
    pf = pareto_front(X_init)
    igd = get_performance_indicator("igd", pf)
    X=pop[0].Position.reshape((1, variables_no))
    print("IGD", igd.do(X))

    return Best_vulture1_individual.Cost, Best_vulture1_X, convergence_curve
