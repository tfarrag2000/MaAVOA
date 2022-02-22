import os
import time
import pickle
import numpy as np
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.factory import get_reference_directions, get_visualization, get_problem, get_termination
from pymoo.indicators.gd import GD
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from MaAVOA_Code.MaAVOA import MaAVOA


def setupFrameWork(algorithmClass, problem, n_obj, termination=None, pop_size=None, runID=1, saveResults=True,maindir=None):
    problemfullname = '{}_obj{}_{}'.format(problem.Name, n_obj, problem.AlgorithmName)
    print(problemfullname + " run_{}".format(runID))

    # n_points = {3: 91, 5: 210, 8: 156, 10: 275, 15:135}
    # ref_dirs = get_reference_directions("energy", Objective_no, n_points[Objective_no], seed=1)
    # reference_direction
    n_partitions = {3: (16, 0), 5: (6, 0), 8: (3, 2), 10: (3, 2), 15: (2, 1), 20: (2, 1), 30: (2, 1)}
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
    # initialization = Initialization(FloatRandomSampling())
    # init_pop = initialization.do(problem, pop_size)

    algorithm = algorithmClass(ref_dirs=ref_dirs)
    # algorithm.setup(problem, termination, seed=1, save_history=False, verbose=False)

    start_time = time.time()
    res = minimize(problem,
                   algorithm,
                   termination,
                   # seed=1,
                   verbose=False,
                   save_history=True)

    exec_time = time.time() - start_time
    # find the pareto_front of the combined results
    F = res.opt.get("F")
    X = res.opt.get("X")
    n_gen = res.algorithm.n_gen
    n_eval = res.algorithm.evaluator.n_eval

    print("done optimization")

    PF = problem.pareto_front(ref_dirs)

    if PF is not None:
        igd = IGD(PF, zero_to_one=True).do(F)
        gd = GD(PF, zero_to_one=True).do(F)
        igdplus = IGDPlus(PF, zero_to_one=True).do(F)
        HV = 0  # HV = Hypervolume(ref_point=np.ones(n_obj)).do(F)
    else:
        pf_dir=os.path.join("D:\OneDrive\My Research\Many_Objectives\The Code\MaAVOA_Code\PF\PlatEmo", "PF_{}_{}.txt".format(problem.Name,n_obj))
        PF=np.genfromtxt(pf_dir, delimiter=',')
        igd = IGD(PF, zero_to_one=True).do(F)
        gd = GD(PF, zero_to_one=True).do(F)
        igdplus = IGDPlus(PF, zero_to_one=True).do(F)
        HV = 0


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

            v= get_visualization("pcp", color="grey", alpha=0.5).add(PF, label="Pareto-front", color="grey",alpha=0.3)
            v.add(F,label="Result",color="blue").save(os.path.join(dir, "{}_PCP.png".format(problemfullname)))

        with open(os.path.join(dir, 'result_object.pkl'), 'wb') as file:
            pickle.dump(res, file)

        #     n_evals = np.array([e.evaluator.n_eval for e in res.history])
        #     opt = np.array([e.opt[0].F for e in res.history])
        #     np.savetxt(os.path.join(dir, "history_n_evals.csv"), n_evals, delimiter=",")
        #     np.savetxt(os.path.join(dir, "history_opt.csv"), opt, delimiter=",")
        #
        #     plt.title("Convergence")
        #     plt.plot(n_evals, opt, "--")
        #     plt.yscale("log")
        #     plt.savefig(os.path.join(dir, 'Convergence.png'))

        with open(os.path.join(dir, "final_result_run.csv"), 'w') as file:
            file.write(
                "problem_Name, n_obj, AlgorithmName, n_gen,n_eval, pop_size, exec_time, igd, gd, igdplus, HV\n")
            file.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {},{}\n".format(problem.Name,
                                                                            n_obj, problem.AlgorithmName,
                                                                            n_gen, n_eval, res.pop.shape[0],
                                                                            round(exec_time, 3),
                                                                            igd, gd, igdplus, HV))
        print("done saving to {}".format(dir))
    print(igd, HV, igdplus)
    print("=" * 100)

    return igd, gd


if __name__ == '__main__':
    ALGORITHMS = [("MaAVOA_70_90", MaAVOA), ("nsga3", NSGA3), ("unsga3", UNSGA3), ("moead", MOEAD),
                  ("ctaea", CTAEA)]
    termination = get_termination("n_eval", 100000) # run 2
    # termination = get_termination("n_gen", 500) # run 1
    # termination = get_termination("time", "00:00:30")  # run 3
    i=0
    for runId in [23,24,25]:
        for n_obj in [3,5,8,10,15]:
            for pID in [1,2,3,4,5,6,7]:  # dtlz
                for alg, algorithmClass in ALGORITHMS:
                    k = 10
                    if pID == 1:
                        k = 5
                    if pID == 7:
                        k = 20
                    n_var = n_obj + k - 1
                    problem_name = "dtlz{}".format(pID)
                    problem = get_problem(problem_name, n_var=n_var, n_obj=n_obj)
                    problem.Name = problem_name
                    problem.AlgorithmName = alg

                    maindir = r'C:\Many_Objectives\DTLZ_Problems'
                    problemfullname = '{}_obj{}_{}'.format(problem.Name, n_obj, problem.AlgorithmName)
                    # dir = os.path.join(maindir, '{}\\run_{}\\result_object.pkl'.format(problemfullname, runId))

                    dir = os.path.join(maindir, '{}\\run_{}\\final_result_run.csv'.format(problemfullname, runId))
                    i = i + 1
                    if os.path.exists(dir):
                        print("{}- {}  done".format(i,problemfullname))
                        continue
                    setupFrameWork(algorithmClass, problem, n_obj, termination=termination, runID=runId,
                                   saveResults=True,maindir=maindir)
