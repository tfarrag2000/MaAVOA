import os
import time

import numpy as np
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.factory import get_reference_directions, get_visualization, get_problem, get_termination
from pymoo.indicators.gd import GD
from pymoo.optimize import minimize

from AVOA_ManyObjectives.MaAVOA_v2 import MaAVOA_v2
from old_trials.IGD import IGD


def setupFrameWork(algorithmClass, problem, Objective_no, termination=None, pop_size=None, runID=1, saveResults=True):
    problemfullname = '{}_obj{}_{}'.format(problem.Name, Objective_no, problem.AlgorithmName)
    print(problemfullname)

    # n_points = {3: 91, 5: 210, 8: 156, 10: 275, 15:135}
    # ref_dirs = get_reference_directions("energy", Objective_no, n_points[Objective_no], seed=1)
    # reference_direction
    n_partitions = {3: (16, 0), 5: (6, 0), 8: (3, 2), 10: (3, 2), 15: (2, 1), 20: (2, 1), 30: (2, 1)}
    ref_dirs = None
    p1 = n_partitions[Objective_no][0]
    p2 = n_partitions[Objective_no][1]
    if p1 != 0:
        ref_dirs_L1 = get_reference_directions("das-dennis", n_dim=Objective_no, n_partitions=p1)
        ref_dirs = ref_dirs_L1
    if p2 != 0:
        ref_dirs_L2 = get_reference_directions("das-dennis", n_dim=Objective_no, n_partitions=p2)
        ref_dirs = np.concatenate((ref_dirs, ref_dirs_L2))

    if pop_size == None:
        pop_size = len(ref_dirs)

    PF = problem.pareto_front(ref_dirs)
    if termination == None:
        termination = get_termination("n_eval", 100000)
    # initialization = Initialization(FloatRandomSampling())
    # init_pop = initialization.do(problem, pop_size)

    algorithm = algorithmClass(ref_dirs=ref_dirs, pop_size=5)
    # algorithm.setup(problem, termination, seed=1, save_history=False, verbose=False)

    start_time = time.time()
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   verbose=True)

    exec_time = time.time() - start_time
    # find the pareto_front of the combined results
    F = res.opt.get("F")
    X = res.opt.get("X")
    igd = IGD(PF, F)
    gd = GD(PF, zero_to_one=True).do(F)
    HV = 0  # Hypervolume(ref_point=PF).do(F)
    gdPlus = 0  # GDPlus(pf=PF).do(F)
    n_gen = res.algorithm.n_gen
    exec_time = res.exec_time
    if saveResults:
        maindir = r'D:\OneDrive\My Research\Many_Objectives\The Code\AVOA_ManyObjectives\results'

        dir = os.path.join(maindir, '{}\\run_{}\\'.format(problemfullname, runID))
        os.makedirs(dir, exist_ok=True)
        if Objective_no <= 5:
            get_visualization("scatter", angle=(45, 45)).add(PF, label="Pareto-front").add(F, label="Result").save(
                os.path.join(dir, 'result.png'))

        get_visualization("pcp", color="grey", alpha=0.5).add(PF, label="Pareto-front", color="grey", alpha=0.3).add(F,
                                                                                                                     label="Result",
                                                                                                                     color="blue").save(
            os.path.join(dir, 'pcp.png'))
        # from pymoo.visualization.pcp import PCP
        # plot = PCP()
        # plot.set_axis_style(color="grey", alpha=0.5)
        # plot.add(PF, color="grey", alpha=0.3)
        # plot.add(F, color="blue")
        # plot.show()

        with open(os.path.join(dir, "F.csv"), 'w') as file:
            for f in F:
                file.write(np.array2string(f, precision=6, separator=',',
                                           suppress_small=True) + "\n")
        with open(os.path.join(dir, "PF.csv"), 'w') as file:
            for f in PF:
                file.write(np.array2string(f, precision=6, separator=',',
                                           suppress_small=True) + "\n")
        with open(os.path.join(dir, "X.csv"), 'w') as file:
            for x in X:
                file.write(np.array2string(x, precision=6, separator=',',
                                           suppress_small=True) + "\n")

        with open(os.path.join(dir, "final_result_run.csv"), 'w') as file:
            file.write(
                "problem_Name, Obj_no, AlgorithmName, n_gen, pop_size, exec_time,igd, gd, HV, gdPlus\n")
            file.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(problem.Name,
                                                                         Objective_no, problem.AlgorithmName,
                                                                         n_gen + 1, res.pop.shape[0],
                                                                         round(exec_time, 3),
                                                                         igd, gd, HV, gdPlus))

    print(igd, gd)
    return igd, gd


if __name__ == '__main__':
    # s = "NSGA3_das_dennis"  # MOEAD, ("rnsga3", RNSGA3)
    ALGORITHMS = [("maavoa", MaAVOA_v2), ("nsga3", NSGA3), ("unsga3", UNSGA3), ("moead", MOEAD),
                  ("ctaea", CTAEA)]
    # ALGORITHMS = [("nsga3_das_IGD30000", NSGA3)]

    ALGORITHMS = [("MaAVOA", MaAVOA_v2)]
    termination = get_termination("n_gen", 190)

    for runId in range(1, 2):
        for Objective_no in [3, ]:
            for pID in [1]:  # dtlz
                for alg, algorithmClass in ALGORITHMS:
                    k = 10
                    if (pID == 1):
                        k = 5
                    variables_no = Objective_no + k - 1
                    problem_name = "dtlz{}".format(pID)
                    problem = get_problem(problem_name, n_var=variables_no, n_obj=Objective_no)
                    problem.Name = problem_name
                    problem.AlgorithmName = alg
                    setupFrameWork(algorithmClass, problem, Objective_no, termination=termination, runID=runId,
                                   saveResults=True)
