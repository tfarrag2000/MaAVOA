import math
import random
from copy import deepcopy
import numpy as np

from AVOA_ManyObjectives.IGD import calculateigd
from AVOA_ManyObjectives.boundaryCheck import boundaryCheck
from AVOA_ManyObjectives.NonDominatedSorting.EvaluatePopulation import evaluatePopulation
from exploitation import exploitation
from exploration import exploration
from initialization import initialization
from random_select import random_select
from pymoo.visualization.scatter import Scatter


def AVOA(pop_size, max_iter, lower_bound, upper_bound, variables_no, Objective_no,benchmarkFn):

    X = initialization(pop_size, variables_no, upper_bound, lower_bound)
    X_init = deepcopy(X)
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
    ############ IGD ############
    truepf = np.array(load_truePF(Objective_no))

    D_Position = []
    D_Cost = []

    pop_CD, F_CD, pop_obj, F_obj = evaluatePopulation(X, pop_size, benchmark=benchmarkFn,AccordingTo=-1)
    D_Position.extend([pop_obj[x].X for x in F_obj[0]])
    D_Cost.extend([pop_obj[x].F for x in F_obj[0]])
    pop = pop_obj
    F_Rank = F_obj

    while current_iter < max_iter:

        X_old = deepcopy(X)

        ################### select V(V) ###################
        Best_V1_id = random.choice(F_Rank[0])
        if len(F_Rank) == 1:
            Best_V2_id = random.choice(F_Rank[0])
        else:
            Best_V2_id = random.choice(F_Rank[1])
        Best_V1_individual = pop[Best_V1_id]
        Best_V2_individual = pop[Best_V2_id]
        Best_V1_X = Best_V1_individual.X.reshape((1, variables_no))
        Best_V2_X = Best_V2_individual.X.reshape((1, variables_no))

        ################### Africian exploration & exploitation ###################
        a = np.random.uniform(- 2, 2, (1, 1)) * ((np.sin((math.pi / 2) * (current_iter / max_iter)) ** gamma) + np.cos((math.pi / 2) * (current_iter / max_iter)) - 1)
        P1 = (2 * np.random.rand() + 1) * (1 - (current_iter / max_iter)) + a
        # Update the location
        for i in range(X.shape[0]):
            current_V_X = X[i, :]
            F = P1 * (2 * np.random.rand() - 1)
            random_V_X = random_select(current_V_X, Best_V1_X, Best_V2_X)
            if np.abs(F) >= 1:
                current_V_X = exploration(current_V_X, random_V_X, F,  p1, upper_bound, lower_bound)
            else:
                if np.abs(F) < 1:
                    current_V_X = exploitation(current_V_X, Best_V1_X, Best_V2_X,
                                               random_V_X, F, p2, p3, variables_no, upper_bound,
                                               lower_bound)
            X[i, :] = current_V_X
        convergence_curve.append(Best_V1_individual.Cost[1])
        current_iter = current_iter + 1
        X_new = boundaryCheck(X, lower_bound, upper_bound)

        ##########################################################################
        X_intermediate = np.concatenate([X_old, X_new])

        pop_CD, F_CD, pop_obj, F_obj = evaluatePopulation(X_intermediate, pop_size)
        pf = np.array([pop_obj[x].F for x in F_obj[0]])
        print("IGD", calculateigd(truepf, pf))
        D_Position.extend([pop_obj[x].X for x in F_obj[0]])
        D_Cost.extend([pop_obj[x].F for x in F_obj[0]])

        X = np.array([p.X for p in pop_CD])
        print('In Iteration %d, best estimation of Conv. and Div. is %4.2f , %4.2f' % (
            current_iter, Best_V1_individual.Cost[0], Best_V1_individual.Cost[1]))

        pop_CD, F_CD, pop_obj, F_obj = evaluatePopulation(np.array(D_Position).reshape((-1, variables_no)), pop_size)
        # to be used in selecting v1 ,v2
        pop = pop_obj
        F_Rank = F_obj

    #################### Order final List
    Archive=np.array(D_Position).reshape((-1, variables_no))
    Archive_CostObj= np.array(D_Cost).reshape((-1, Objective_no))
    np.savetxt('Archive.txt', Archive, delimiter=',')
    np.savetxt('Archive_CostObj.txt', Archive_CostObj, delimiter=',')

    pop_CD, F_CD, pop_obj, F_obj = evaluatePopulation(Archive, pop_size, variables_no, Objective_no)
    X_list = np.array([pop_obj[x].X for x in F_obj[0]]).reshape((-1, variables_no))
    pf = np.array([pop_obj[x].F for x in F_obj[0]]).reshape((-1, Objective_no))

    np.savetxt('pf.txt', X_list, delimiter=',')
    np.savetxt('pf_CostObj.txt', pf, delimiter=',')

    print("IGD_Final", calculateigd(truepf, pf))

    # Scatter(legend=True).add(pf, label="Pareto-front").add(X_Pareto_Front, label="Result").show()

    return Best_V1_individual.Cost, Best_V1_X, convergence_curve


def load_truePF(Objective_no):
    mainlist = []
    infile = open('PF_{}.txt'.format(Objective_no), 'r')
    for line in infile:
        list1 = line.strip().split(' ')
        list2 = [float(i) for i in list1]
        mainlist.append(list2)
    infile.close()
    return mainlist
