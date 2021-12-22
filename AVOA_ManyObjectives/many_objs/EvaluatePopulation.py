import random

import numpy as np

from AVOA_ManyObjectives.many_objs.NonDominatedSorting import NonDominatedSorting
from AVOA_ManyObjectives.many_objs.PlotCosts import PlotCosts
from AVOA_ManyObjectives.many_objs.SortPopulation import SortPopulation
from AVOA_ManyObjectives.many_objs.benchmark import benchmark
from AVOA_ManyObjectives.many_objs.empty_individual import empty_individual


def init_pop():
    nPop = 100
    nobj = 3

    VarMin = np.array([0, 0, 0])
    VarMax = np.array([5, 5, 5])
    nVars = np.asarray(VarMin).size
    Ranges = VarMax - VarMin
    ### intialize random npop individual
    X = np.zeros((nPop, nobj))
    for i in range(nPop):
        xRand = VarMin + np.multiply(np.random.rand(1, nVars), Ranges)
        X[i, :] = xRand
    # mat = scipy.io.loadmat('Matlab_vr/X.mat')
    # X = mat['X']

    pop, F = evaluatePopulation(X)

    F1 = [pop[i] for i in F[0]]
    PlotCosts(F1)

    x = np.array([1, np.asarray(F[0]).size])
    RS = random.randint(0, np.asarray(F[0]).size - 1)
    uide_sol = pop[RS].Position


def evaluatePopulation(X, n, variables_no, Objective_no):
    nPop = X.shape[0]
    # ### calculate objactive function
    objmatrix = benchmark(X, variables_no, Objective_no)

    # #### ideal point
    # Zideal = np.amin(objmatrix, axis=0)
    # #### nadir point
    # Znadir = np.amax(objmatrix, axis=0)
    # c1 = np.zeros((nPop, 1))
    # c2 = np.zeros((nPop, 1))
    # for i in range(nPop):
    #     c1[i] = np.sqrt(sum(objmatrix[i, :] - Zideal) ** 2)
    #     c2[i] = np.sqrt(sum(objmatrix[i, :] - Znadir) ** 2)
    #
    # C = np.concatenate((c1, c2), axis=1)
    # Cnadir = np.amax(C, axis=0)
    #
    # conve = np.zeros((nPop, 1))
    # for i in range(nPop):
    #     conve[i] = LA.norm(C[i, :] - Cnadir)
    #
    # A = np.zeros((nPop, nPop))
    #
    # for i in range(nPop):
    #     for j in range(nPop):
    #         if i == j:
    #             A[i, j] = 10
    #         else:
    #             A[i, j] = np.arccos(sum(np.multiply((objmatrix[i, :] - Zideal), (objmatrix[j, :] - Zideal))) / (
    #                     (np.sqrt(sum(objmatrix[i, :] - Zideal) ** 2)) * (np.sqrt(sum(objmatrix[j, :] - Zideal) ** 2))))
    #
    # Dive = np.amin(A, axis=0).reshape((-1, 1))
    # perfermancmetric = np.concatenate((conve, Dive), axis=1) * - 1

    pop = []
    for i in range(nPop):
        individual = empty_individual()
        individual.Position = X[i, :]
        # individual.Cost = perfermancmetric[i, :]
        individual.Cost = objmatrix[i, :]
        pop.append(individual)

    pop, F = NonDominatedSorting(pop, n)
    # Calculate Crowding Distance
    # pop = CalcCrowdingDistance(pop, F)
    # Sort Population
    pop, F = SortPopulation(pop, n)

    return pop, F


if __name__ == '__main__':
    init_pop()
