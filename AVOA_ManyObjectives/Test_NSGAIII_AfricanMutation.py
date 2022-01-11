import os
import time

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.initialization import Initialization
from pymoo.factory import get_problem, get_reference_directions, get_visualization
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.termination.no_termination import NoTermination

from AVOA_ManyObjectives.AfricanMutation import AfricanMutation


def setupFrameWork(problem, pop_size, Objective_no, generation_no, runID=1, saveResults=True):
    problemfullname = '{}_obj{}_{}'.format(problem.Name, Objective_no, problem.AlgorithmName)
    print(problemfullname)

    n_points = {3: 92, 5: 212, 8: 156, 10: 112, 15: 136}
    ref_dirs = get_reference_directions("energy", Objective_no, n_points[Objective_no], seed=1)
    PF = problem.pareto_front(ref_dirs)
    termination = NoTermination()
    initialization = Initialization(FloatRandomSampling())
    init_pop = initialization.do(problem, pop_size)

    algorithm = NSGA3(pop_size=len(init_pop), ref_dirs=ref_dirs,
                      mutation=AfricanMutation(eta=20, ref_dirs=ref_dirs, prob=0.5),
                      sampling=init_pop)
    algorithm.setup(problem, termination, seed=1, save_history=False, verbose=False)
    # until the algorithm has no terminated
    start_time = time.time()
    for n_gen in range(generation_no):
        print(n_gen)
        # ask the algorithm for the next solution to be evaluated
        pop = algorithm.ask()
        algorithm.evaluator.eval(problem, pop)
        algorithm.tell(infills=pop)

    exec_time = time.time() - start_time

    # find the pareto_front of the combined results
    F = algorithm.survival.opt.get("F")
    X = algorithm.survival.opt.get("X")
    igd = IGD(PF, zero_to_one=True).do(F)
    gd = GD(PF, zero_to_one=True).do(F)
    # HV = Hypervolume(pf=PF).do(F)
    # gdPlus=GDPlus(pf=PF).do(F)
    if saveResults:
        maindir = r'D:\OneDrive\My Research\Many_Objectives\The Code\AVOA_ManyObjectives\results\NSGAIII_AfricanMutation'

        dir = os.path.join(maindir, '{}\\run_{}\\'.format(problemfullname, runID))
        os.makedirs(dir, exist_ok=True)
        if Objective_no <= 5:
            get_visualization("scatter", angle=(45, 45)).add(PF, label="Pareto-front").add(F, label="Result").save(
                os.path.join(dir, 'result.png'))

        with open(os.path.join(dir, "F.csv"), 'w') as file:
            for f in F:
                file.write(np.array2string(f, precision=6, separator=',',
                                           suppress_small=True) + "\n")
        with open(os.path.join(dir, "X.csv"), 'w') as file:
            for x in X:
                file.write(np.array2string(x, precision=6, separator=',',
                                           suppress_small=True) + "\n")

        with open(os.path.join(dir, "final_result_run.csv"), 'w') as file:
            file.write("n_gen, Obj_no ,pop_size ,igd, gd, exec_time\n")
            file.write("{}, {}, {}, {}, {}, {}\n".format(n_gen + 1, Objective_no, pop_size, igd, gd, exec_time))

    print(n_gen, igd, gd)
    return n_gen, igd, gd


if __name__ == '__main__':
    generation_no = 200
    pop_size = 1000
    for runId in range(1, 2):
        for Objective_no in [3, 5, 8, 10]:
            for pID in [1, 2, 3, 4]:  # dtlz
                k = 5
                variables_no = Objective_no + k - 1
                problem_name = "dtlz{}".format(pID)

                problem = get_problem(problem_name, n_var=variables_no, n_obj=Objective_no)
                problem.Name = problem_name
                problem.AlgorithmName = "NSGAIII_AfricanMutatio"

                setupFrameWork(problem, pop_size, Objective_no, generation_no, runId)
