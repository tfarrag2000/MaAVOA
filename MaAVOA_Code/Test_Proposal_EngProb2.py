import os
import pickle

import numpy as np
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.factory import get_reference_directions, get_visualization, get_termination, get_sampling, get_crossover, \
    get_mutation
from pymoo.indicators.gd import GD
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.optimize import minimize

from EngProblems.EngProb2 import EngProb2
from MaAVOA_Code.MaAVOA_Mix import MaAVOA_Mix


def setupFrameWork(algorithmClass, problem, n_obj, termination=None, pop_size=None, runID=1, saveResults=True):
    problemfullname = '{}_obj{}_{}'.format(problem.Name, n_obj, problem.AlgorithmName)
    print(problemfullname + " run id: {}".format(runID))

    # n_points = {3: 91, 5: 210, 8: 156, 10: 275, 15:135}
    # ref_dirs = get_reference_directions("energy", Objective_no, n_points[Objective_no], seed=1)
    # reference_direction
    n_partitions = {3: (16, 0), 4: (16, 0), 5: (6, 0), 8: (3, 2), 10: (3, 2), 15: (2, 1), 20: (2, 1), 30: (2, 1)}
    ref_dirs = None
    p1 = n_partitions[n_obj][0]
    p2 = n_partitions[n_obj][1]
    if p1 != 0:
        ref_dirs_L1 = get_reference_directions("das-dennis", n_dim=n_obj, n_partitions=p1)
        ref_dirs = ref_dirs_L1
    if p2 != 0:
        ref_dirs_L2 = get_reference_directions("das-dennis", n_dim=n_obj, n_partitions=p2)
        ref_dirs = np.concatenate((ref_dirs, ref_dirs_L2))

    if pop_size == None:
        pop_size = len(ref_dirs)

    # np.savetxt('ref_dirs_{}.txt'.format(Objective_no,len(ref_dirs)), ref_dirs, delimiter=',')
    # np.savetxt('PF_{}_{}.txt'.format(problem.Name,n_obj), PF, delimiter=',')

    if termination == None:
        termination = get_termination("n_eval", 100000)
    # init_pop = initialization.do(problem, pop_size)
    mask = ["real", "real", "real", "real", "int", "int", "int", "int"]

    sampling = MixedVariableSampling(mask, {"real": get_sampling("real_random"), "int": get_sampling("int_random")})
    crossover = MixedVariableCrossover(mask, {
        "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
        "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
    })

    mutation = MixedVariableMutation(mask, {
        "real": get_mutation("real_pm", eta=3.0),
        "int": get_mutation("int_pm", eta=3.0)
    })
    algorithm = algorithmClass(ref_dirs=ref_dirs, sampling=sampling,
                               crossover=crossover,
                               mutation=mutation,
                               eliminate_duplicates=True)
    # algorithm.setup(problem, termination, seed=1, save_history=False, verbose=False)

    res = minimize(problem,
                   algorithm,
                   termination,
                   n_constr=3,
                   # seed=1,
                   verbose=False,
                   save_history=False)

    # find the pareto_front of the combined results
    F = res.opt.get("F")
    X = res.opt.get("X")
    n_gen = res.algorithm.n_gen
    n_eval = res.algorithm.evaluator.n_eval

    print("done optimization")
    PF = np.genfromtxt(".\\PF\\PF_EngProb1_4.txt",
                       delimiter=',')

    HV = Hypervolume(ref_point=np.ones(n_obj)).do(F)

    if PF is not None:
        igd = IGD(PF, zero_to_one=True).do(F)
        gd = GD(PF, zero_to_one=True).do(F)
        igdplus = IGDPlus(PF, zero_to_one=True).do(F)
    else:
        igd = -1
        gd = -1
        igdplus = -1

    if saveResults:
        dir = os.path.join(maindir, '{}\\run_{}\\'.format(problemfullname, runID))
        os.makedirs(dir, exist_ok=True)

        np.savetxt(os.path.join(dir, "F_new.csv"), F, delimiter=",")
        np.savetxt(os.path.join(dir, "X_new.csv"), X, delimiter=",")
        if PF is not None:
            np.savetxt(os.path.join(dir, "PF_new.csv"), PF, delimiter=",")
            if n_obj <= 5:
                get_visualization("scatter", angle=(45, 45)).add(PF, label="Pareto-front").add(F, label="Result").save(
                    os.path.join(dir, "{}_3D_Scatter.png".format(problemfullname)))

            v = get_visualization("pcp", color="grey", alpha=0.5).add(PF, label="Pareto-front", color="grey", alpha=0.3)
            v.add(F, label="Result", color="blue").save(os.path.join(dir, "{}_PCP.png".format(problemfullname)))

        with open(os.path.join(dir, 'result_object.pkl'), 'wb') as file:
            pickle.dump(res, file)

        with open(os.path.join(dir, "final_result_run.csv"), 'w') as file:
            file.write(
                "problem_Name, n_obj, AlgorithmName, n_gen,n_eval, pop_size, exec_time, igd, gd, igdplus, HV\n")
            file.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {},{}\n".format(problem.Name,
                                                                            n_obj, problem.AlgorithmName,
                                                                            n_gen, n_eval, res.pop.shape[0],
                                                                            round(res.exec_time, 3),
                                                                            igd, gd, igdplus, HV))
        print("done saving to {}".format(dir))
    print("Pareto Front len :{}".format(len(F)))
    print("=" * 100)


if __name__ == '__main__':
    ALGORITHMS = [("MaAVOA_70_90", MaAVOA_Mix), ("nsga3", NSGA3), ("unsga3", UNSGA3),  # ("moead", MOEAD),
                  ("ctaea", CTAEA), ("AGEMOEA", AGEMOEA), ("RNSGA3", RNSGA3)]

    # termination = get_termination("time", "00:00:30")
    # termination = get_termination("n_eval", 100000)

    i = 0
    for alg, algorithmClass in ALGORITHMS:
        for n_gen in [250, 500, 1000,2000,4000,5000,10000]:
            try:
                termination = get_termination("n_gen", n_gen)
                problem_name = "EngProb2"
                problem = EngProb2()
                problem.Name = problem_name
                problem.AlgorithmName = alg
                maindir = r'C:\Many_Objectives\EngProblem2'
                problemfullname = '{}_obj{}_{}'.format(problem.Name, 4, problem.AlgorithmName)
                dir = os.path.join(maindir, '{}\\run_{}\\final_result_run.csv'.format(problemfullname, n_gen))
                i = i + 1
                if os.path.exists(dir):
                    print("{}- {} -- run id:{} done".format(i, problemfullname, n_gen))
                    continue

                setupFrameWork(algorithmClass, problem, 4, termination=termination, runID=n_gen,
                               saveResults=True)
            except Exception as e:
                print("{}- {} -- run id:{} Erroooor:{}".format(i, problemfullname, n_gen, e))
