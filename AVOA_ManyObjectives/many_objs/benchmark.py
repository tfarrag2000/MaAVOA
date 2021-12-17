import pymoo.problems.many.dtlz as dtlz
from pymoo.factory import get_problem


def benchmark(X):
    dtlz_problem = dtlz.DTLZ1(n_var=3, n_obj=3)
    g1 = dtlz_problem.g1(X)
    g2 = dtlz_problem.g2(X)
    objmatrix = dtlz_problem.obj_func(X, g1)

    return objmatrix


def pareto_front(X):
    pf = get_problem("dtlz1").pareto_front(X)
    return pf
    # get_visualization("scatter", angle=(45, 45)).add(pf).show()
