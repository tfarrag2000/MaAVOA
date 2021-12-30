from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions

from AVOA_ManyObjectives.AfricanMutation import AfricanMutation

problem = get_problem("dtlz1", n_var=7, n_obj=3)
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
PF = get_problem("dtlz1", n_var=7, n_obj=3).pareto_front(ref_dirs)

# Scatter().add(PF).show()

# create the algorithm object
algorithm1 = NSGA3(pop_size=100, ref_dirs=ref_dirs)
algorithm1.setup(problem, ('n_gen', 10), seed=1, save_history=True, verbose=True)

algorithm2 = NSGA3(pop_size=100, ref_dirs=ref_dirs)
algorithm2.setup(problem, ('n_gen', 10), mutation=AfricanMutation(eta=20, prob=0.8),seed=1, save_history=True, verbose=True)


# until the algorithm has no terminated
while algorithm1.has_next():
    # ask the algorithm for the next solution to be evaluated
    algorithm1.next()
    pop = algorithm1.ask()
    # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
    algorithm1.evaluator.eval(problem, pop)
    # returned the evaluated individuals which have been evaluated or even modified
    algorithm1.tell(infills=pop)
    # do same more things, printing, logging, storing or even modifying the algorithm object
    print(algorithm1.n_gen, algorithm1.evaluator.n_eval)


# obtain the result objective from the algorithm
res = algorithm1.result()

# calculate a hash to show that all executions end with the same result
print("hash", res.F.sum())
