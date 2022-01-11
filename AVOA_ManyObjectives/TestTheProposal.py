import os
import time
from collections import Counter

import numpy as np
from fcmeans import FCM
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population, pop_from_array_or_individual
from pymoo.factory import get_reference_directions, get_visualization, get_problem
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.termination.no_termination import NoTermination

from MaAVOA import MaAVOA


def setupFrameWork(problem, pop_size, Objective_no, generation_no, runID=1, saveResults=True, n_clusters=3,
                   Combined_every=1):
    problemfullname = '{}_obj{}_{}_{}_{}'.format(problem.Name, Objective_no, problem.AlgorithmName, n_clusters,
                                                 Combined_every)
    print(problemfullname)

    n_points = {3: 92, 5: 212, 8: 156, 10: 112, 15: 136}
    ref_dirs = get_reference_directions("energy", Objective_no, n_points[Objective_no], seed=1)
    PF = problem.pareto_front(ref_dirs)
    termination = NoTermination()
    initialization = Initialization(FloatRandomSampling())
    init_pop = initialization.do(problem, pop_size)

    list_init_pop = do_Clustring(init_pop, n_clusters=n_clusters)

    algorithms = []

    for i in range(n_clusters):
        algorithm = MaAVOA(pop_size=len(list_init_pop[i]), ref_dirs=ref_dirs, sampling=np.array(list_init_pop[i]))
        algorithm.setup(problem, termination, seed=1, save_history=False, verbose=False)
        algorithms.append(algorithm)

    start_time = time.time()
    c_e = 0;

    for n_gen in range(generation_no):
        print(n_gen)
        # ask the algorithm for the next solution to be evaluated
        pop_list = []
        for algorithm in algorithms:
            runAlgorithm(algorithm, problem, pop_list)
            algorithm.n_gen = n_gen

        # Processes=[]
        # pop_list=multiprocessing.Manager().list()
        # for algorithm in algorithms:
        #     p = Process(target=runAlgorithm, args=(algorithm,problem,pop_list ) ,)
        #     p.start()
        #     Processes.append(p)
        #     #
        # for j in Processes:
        #     j.join()
        c_e += 1
        # combined results and reclustering
        if (c_e == Combined_every and n_clusters!=1):
            new_init_pop = Population()
            for p in pop_list:
                new_init_pop = Population.merge(new_init_pop, p)
            list_init_pop = do_Clustring(new_init_pop, n_clusters=len(algorithms))
            for algorithm, pop in zip(algorithms, list_init_pop):
                algorithm.infills = pop_from_array_or_individual(pop)
            c_e = 0
    #################

    final_pop = Population()
    for p in pop_list:
        new_init_pop = Population.merge(final_pop, p)
    exec_time = time.time() - start_time
    survival = ReferenceDirectionSurvival(ref_dirs)
    pop = survival.do(problem, new_init_pop)

    # find the pareto_front of the combined results
    F = survival.opt.get("F")
    X = survival.opt.get("X")
    igd = IGD(PF, zero_to_one=True).do(F)
    gd = GD(PF, zero_to_one=True).do(F)
    HV = Hypervolume(pf=PF).do(F)
    gdPlus=GDPlus(pf=PF).do(F)
    if saveResults:
        maindir = r'D:\OneDrive\My Research\Many_Objectives\The Code\AVOA_ManyObjectives\results'

        dir = os.path.join(maindir, '{}\\run_{}\\'.format(problemfullname, runID))
        os.makedirs(dir, exist_ok=True)
        if Objective_no <= 5:
            get_visualization("scatter", angle=(45, 45)).add(PF, label="Pareto-front").add(F, label="Result").save(
                os.path.join(dir, 'result.png'))

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
        with open(os.path.join(dir, "X.csv"), 'w') as file:
            for x in X:
                file.write(np.array2string(x, precision=6, separator=',',
                                           suppress_small=True) + "\n")

        with open(os.path.join(dir, "final_result_run.csv"), 'w') as file:
            file.write("n_gen, Obj_no ,pop_size ,igd, gd, exec_time\n")
            file.write("{}, {}, {}, {}, {}, {}\n".format(n_gen + 1, Objective_no, pop_size, igd, gd, exec_time))

    print(n_gen, igd, gd)
    return n_gen, igd, gd


def runAlgorithm(algorithm, problem, pop_list):
    pop = algorithm.ask()
    algorithm.evaluator.eval(problem, pop)
    algorithm.tell(infills=pop)
    pop_list.append(pop)


def do_Clustring(init_pop, n_clusters):
    if isinstance(init_pop, Population):
        X = init_pop.get("X")

    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(X)
    # outputs
    fcm_centers = fcm.centers
    fcm_labels = fcm.predict(X)
    fcm_stat = Counter(fcm_labels)
    init_pop.set("Cluster", fcm_labels)
    init_pop.set("Data", None)
    list_pop = []
    for i in range(n_clusters):
        list_pop.append([])

    for p in init_pop:
        list_pop[p.data["Cluster"]].append(p.X)

    return list_pop


if __name__ == '__main__':
    generation_no = 500
    Combined_every = 100
    pop_size = 1000
    n_clusters = 1

    s = "MaAVOA"

    for runId in range(1, 4):
        for Objective_no in [3]:
            for pID in [1]:  # dtlz
                k = 5
                variables_no = Objective_no + k - 1
                problem_name = "dtlz{}".format(pID)

                problem = get_problem(problem_name, n_var=variables_no, n_obj=Objective_no)
                problem.Name = problem_name
                problem.AlgorithmName = s

                setupFrameWork(problem, pop_size, Objective_no, generation_no, runId, saveResults=True,
                               n_clusters=n_clusters, Combined_every=Combined_every)
