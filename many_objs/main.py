import random

import numpy as np
from numpy import linalg as LA

from CalcCrowdingDistance import CalcCrowdingDistance
from NonDominatedSorting import NonDominatedSorting
from PlotCosts import PlotCosts
from SortPopulation import SortPopulation
from empty_individual import empty_individual
from many_objs.benchmark import benchmark

nPop = 100


def init_pop():
    VarMin = np.array([0, 0, 0])
    VarMax = np.array([5, 5, 5])
    nobj = 3
    nVars = np.asarray(VarMin).size
    Ranges = VarMax - VarMin
    ### intialize random npop individual
    X = np.zeros((nPop, nobj))
    for i in range(nPop):
        xRand = VarMin + np.multiply(np.random.rand(1, nVars), Ranges)
        X[i, :] = xRand
    # mat = scipy.io.loadmat('Matlab_vr/X.mat')
    # X = mat['X']



    pop = getEvaluatedPopulation(X)
    pop, F = NonDominatedSorting(pop)
    # Calculate Crowding Distance
    pop = CalcCrowdingDistance(pop, F)
    # Sort Population
    pop, F = SortPopulation(pop)

    F1 = [pop[i] for i in F[0]]
    PlotCosts(F1)

    x = np.array([1, np.asarray(F[0]).size])
    RS = random.randint(0, np.asarray(F[0]).size - 1)
    uide_sol = pop[RS].Position



def getEvaluatedPopulation(X):
    # ### calculate objactive function
    objmatrix= benchmark(X)

    #### ideal point
    Zideal = np.amin(objmatrix, axis=0)
    #### nadir point
    Znadir = np.amax(objmatrix, axis=0)
    c1 = np.zeros((nPop, 1))
    c2 = np.zeros((nPop, 1))
    for i in range(nPop):
        c1[i] = np.sqrt(sum(objmatrix[i, :] - Zideal) ** 2)
        c2[i] = np.sqrt(sum(objmatrix[i, :] - Znadir) ** 2)

    C = np.concatenate((c1, c2), axis=1)
    Cnadir = np.amax(C, axis=0)

    conve = np.zeros((nPop, 1))
    for i in range(nPop):
        conve[i] = LA.norm(C[i, :] - Cnadir)

    A = np.zeros((nPop, nPop))

    for i in range(nPop):
        for j in range(nPop):
            if i == j:
                A[i, j] = 10
            else:
                A[i, j] = np.arccos(sum(np.multiply((objmatrix[i, :] - Zideal), (objmatrix[j, :] - Zideal))) / (
                        (np.sqrt(sum(objmatrix[i, :] - Zideal) ** 2)) * (np.sqrt(sum(objmatrix[j, :] - Zideal) ** 2))))

    Dive = np.amin(A, axis=0).reshape((-1, 1))
    perfermancmetric = np.concatenate((conve, Dive), axis=1) * - 1

    # mat = scipy.io.loadmat('Matlab_vr/perfermancmetric.mat')
    # perfermancmetric = mat['perfermancmetric']

    pop = []
    for i in range(nPop):
        individual = empty_individual()
        pop.append(individual)

    for i in range(nPop):
        pop[i].Position = X[i, :]
        pop[i].Cost = perfermancmetric[i, :]

    return pop


if __name__ == '__main__':
    init_pop()
