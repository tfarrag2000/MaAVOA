import pymoo.problems.many.dtlz as dtlz


def benchmark(X=None):
    dtlz_problem = dtlz.DTLZ1(n_var=3, n_obj=3)
    g1 = benchmark.g1(X)
    g2 = benchmark.g2(X)
    objmatrix = dtlz_problem.obj_func(X, g1)

    return objmatrix
