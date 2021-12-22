import numpy as np
from deap.benchmarks import dtlz1


def benchmark(X, variables_no, Objective_no):
    # X=np.array([[0.5,0.5 ,0.5 ,0.5, 0.5, 0.5, 0.5]])
    # dtlz_problem = dtlz.DTLZ1(n_var=variables_no, n_obj=Objective_no)
    # g1 = dtlz_problem.g1(X)
    # g2 = dtlz_problem.g2(X)
    # objmatrix = dtlz_problem.obj_func(X, g1)

    objmatrix = np.array([dtlz1(x, Objective_no) for x in X])
    return objmatrix
