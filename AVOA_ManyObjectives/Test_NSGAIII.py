from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


def init_pop():
    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    PF = get_problem("dtlz1",n_var=7, n_obj=3).pareto_front(ref_dirs)
    # Scatter().add(PF).show()

    # create the algorithm object
    algorithm = NSGA3(pop_size=100,
                      ref_dirs=ref_dirs)

    # execute the optimization
    res = minimize(get_problem("dtlz1",n_var=7, n_obj=3),
                   algorithm,
                   seed=1,
                   termination=('n_gen', 400), save_history=True,
                   verbose=False)


    Scatter(legend=True).add(PF, label="Pareto-front").add(res.F, label="Result").show()



if __name__ == '__main__':
    init_pop()
