import pymoo.problems.many.dtlz as dtlz


def benchmark(X):
    dtlz_problem = dtlz.DTLZ1(n_var=3, n_obj=3)
    g1 = dtlz_problem.g1(X)
    g2 = dtlz_problem.g2(X)
    objmatrix = dtlz_problem.obj_func(X, g1)
    return objmatrix
