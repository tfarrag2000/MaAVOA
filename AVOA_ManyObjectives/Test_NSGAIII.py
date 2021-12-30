from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from AVOA_ManyObjectives.AfricanMutation import AfricanMutation
from AVOA_ManyObjectives.MaAVOA import MaAVOA


def init_pop():
    # create the reference directions to be used for the optimization

    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    problem = get_problem("dtlz1", n_var=7, n_obj=3)

    PF =problem.pareto_front(ref_dirs)
    # Scatter().add(PF).show()

    # create the algorithm object
    algorithm1 = MaAVOA(pop_size=1000,
                        ref_dirs=ref_dirs)
    algorithm1.setup(problem, ('n_gen', 1000), seed=1, save_history=False, verbose=True)

    # execute the optimization
    # res = minimize(problem,
    #                algorithm1,
    #                seed=1,
    #                termination=('n_gen', 2), save_history=True,
    #                verbose=True)

    # # until the algorithm has no terminated
    while algorithm1.has_next():
        # do the next iteration
        algorithm1.next()

        # do same more things, printing, logging, storing or even modifying the algorithm object
        # print(algorithm1.n_gen, algorithm1.evaluator.n_eval)

    # obtain the result objective from the algorithm
    res = algorithm1.result()

    Scatter(legend=True).add(PF, label="Pareto-front").add(res.F, label="Result").show()



if __name__ == '__main__':
    init_pop()
