import math
import random
from copy import deepcopy

import numpy as np
from pymoo.factory import get_performance_indicator

from AVOA_ManyObjectives.IGD import calculateigd
from AVOA_ManyObjectives.boundaryCheck import boundaryCheck
from AVOA_ManyObjectives.many_objs.EvaluatePopulation import evaluatePopulation
from exploitation import exploitation
from exploration import exploration
from initialization import initialization
from random_select import random_select


def AVOA(pop_size, max_iter, lower_bound, upper_bound, variables_no, Objective_no):
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
    X_new = []

    ############ IGD ############
    pf = np.array(loadPF(Objective_no))
    igd = get_performance_indicator("igd", pf)

    X_intermediate = X

    D_Position = []
    D_Cost = []

    while current_iter < max_iter:

        pop, F_Rank = evaluatePopulation(X_intermediate, pop_size, variables_no, Objective_no)
        X_list = np.array([pop[x].Position for x in F_Rank[0]])
        X_Pareto_Front = np.array([pop[x].Cost for x in F_Rank[0]])
        # print("IGD", igd.do(X_Pareto_Front))
        print("IGD2", calculateigd(pf, X_Pareto_Front))
        # print("IGD3", IGD(pf, X_Pareto_Front))
        D_Position.append(X_list)
        D_Cost.append(X_Pareto_Front)

        Best_vulture1_id = random.choice(F_Rank[0])
        if len(F_Rank) == 1:
            Best_vulture2_id = random.choice(F_Rank[0])
        else:
            Best_vulture2_id = random.choice(F_Rank[1])
        Best_vulture1_individual = pop[Best_vulture1_id]
        Best_vulture2_individual = pop[Best_vulture2_id]
        Best_vulture1_X = Best_vulture1_individual.Position.reshape((1, variables_no))
        Best_vulture2_X = Best_vulture2_individual.Position.reshape((1, variables_no))

        X_new = np.array([p.Position for p in pop])
        X_old = deepcopy(X_new)

        # Africian
        a = np.random.uniform(- 2, 2, (1, 1)) * ((np.sin((math.pi / 2) * (current_iter / max_iter)) ** gamma) + np.cos(
            (math.pi / 2) * (current_iter / max_iter)) - 1)
        P1 = (2 * np.random.rand() + 1) * (1 - (current_iter / max_iter)) + a
        # Update the location
        for i in range(X_new.shape[0]):
            current_vulture_X = X_new[i, :]
            F = P1 * (2 * np.random.rand() - 1)
            random_vulture_X = random_select(current_vulture_X, Best_vulture1_X, Best_vulture2_X)
            if np.abs(F) >= 1:
                current_vulture_X = exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound)
            else:
                if np.abs(F) < 1:
                    current_vulture_X = exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X,
                                                     random_vulture_X, F, p2, p3, variables_no, upper_bound,
                                                     lower_bound)
            X_new[i, :] = current_vulture_X
        convergence_curve.append(Best_vulture1_individual.Cost[1])
        current_iter = current_iter + 1

        X_new = boundaryCheck(X_new, lower_bound, upper_bound)

        X_intermediate = np.concatenate([X_old, X_new])

        print('In Iteration %d, best estimation of Conversion and Diversion is %4.2f , %4.2f' % (
            current_iter, Best_vulture1_individual.Cost[0], Best_vulture1_individual.Cost[1]))

    pop, F_Rank = evaluatePopulation(X_intermediate, pop_size, variables_no, Objective_no)
    X_list = np.array([pop[x].Position for x in F_Rank[0]])
    X_Pareto_Front = np.array([pop[x].Cost for x in F_Rank[0]])
    # print("IGD", igd.do(X_Pareto_Front))
    print("IGD2", calculateigd(pf, X_Pareto_Front))
    # print("IGD3", IGD(pf, X_Pareto_Front))
    D_Position.append(X_list)
    D_Cost.append(X_Pareto_Front)

    # Scatter(legend=True).add(pf, label="Pareto-front").add(X_Pareto_Front, label="Result").show()

    return Best_vulture1_individual.Cost, Best_vulture1_X, convergence_curve


def loadPF(Objective_no):
    mainlist = []
    infile = open('PF_{}.txt'.format(Objective_no), 'r')
    for line in infile:
        list1 = line.strip().split(' ')
        list2 = [float(i) for i in list1]
        mainlist.append(list2)
    infile.close()
    return mainlist
