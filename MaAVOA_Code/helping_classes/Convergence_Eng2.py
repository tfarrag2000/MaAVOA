import os
import pickle

import numpy as np
from pymoo.indicators.igd import IGD

dir = r'C:\Many_Objectives\EngProblem2'

for probname in os.listdir(dir):
    probdir = os.path.join(dir, probname)
    if os.path.isdir(probdir):
        for runname in os.listdir(probdir):
            if runname != "run_1000":
                continue
            print(probname, runname)

            rundir = os.path.join(probdir, runname)
            dir_pkl = os.path.join(rundir, 'result_object.pkl')
            if os.path.exists(dir_pkl):
                if os.path.exists(os.path.join(rundir, "history_igd_list.csv")):
                    print("{}_{}  done".format(probname, runname))
                    continue
                print("{}_{}  start".format(probname, runname))

                file = open(dir_pkl, 'rb')
                res = pickle.load(file)
                file.close()

                PF_filepath = os.path.join(rundir, "PF_new.csv")
                PF = np.genfromtxt(PF_filepath, delimiter=',')

                metric = IGD(PF, zero_to_one=True)
                F_list = [h.opt.get("F") for h in res.history]
                n_evals = np.array([e.evaluator.n_eval for e in res.history])
                igd_list = np.array([metric.do(_F) for _F in F_list])
                np.savetxt(os.path.join(rundir, "history_igd_list.csv"), igd_list, delimiter=",")
                np.savetxt(os.path.join(rundir, "history_n_evals.csv"), n_evals, delimiter=",")
                n_gen = range(1, len(igd_list) + 1)

                # plt.plot(n_gen, igd_list, color='black', lw=0.7, label="Avg. CV of Pop")
                # # plt.scatter(runs, igd_list, facecolor="none", edgecolor='black', marker="p")
                # # plt.axhline(10 ** -2, color="red", label="10^-2", linestyle="--")
                # plt.title("Convergence")
                # plt.xlabel("no. of iterations")
                # plt.ylabel("IGD")
                # plt.yscale("log")
                # plt.legend()
                # plt.savefig(os.path.join(rundir, "conv.png"))
                # plt.cla()
                # plt.clf()
# import os
#
# import numpy as np
# from matplotlib import pyplot as plt
#
# ALGORITHMS = [("MaAVOA_70_90", "MaAVOA"), ("nsga3", "NSGA3"), ("unsga3", "UNSGA3"),("ctaea", "CTAEA")]
# # ALGORITHMS = [("MaAVOA_70_90", "MaAVOA")]
#
#
# for n_obj in [10]:
#     for pID in [ 4]:
#         for alg, Name in ALGORITHMS:
#             problem_name = 'EngProb1_obj4_{}'.format(alg)
#             maindir = r'D:\My Research Results\Many_Objectives\EngProblem1'
#             problemfullname = '{}_obj{}_{}'.format(problem_name, n_obj, alg)
#             print(problemfullname)
#             rundir = os.path.join(maindir, '{}\\run_1000'.format(problemfullname))
#
#             if not os.path.exists(os.path.join(rundir, "history_igd_list.csv")):
#                 continue
#
#             igd_list = np.genfromtxt(os.path.join(rundir, "history_igd_list.csv"), delimiter=',')
#             n_evals = np.genfromtxt(os.path.join(rundir, "history_n_evals.csv"), delimiter=',')
#             n_gen = range(1, len(igd_list) + 1)
#
#             plt.plot(n_gen, igd_list, lw=1, label=Name)
#             # plt.scatter(runs, igd_list, facecolor="none", edgecolor='black', marker="p")
#             # plt.axhline(10 ** -2, color="red", label="10^-2", linestyle="--")
#             plt.title("Convergence")
#             plt.xlabel("no. of iterations")
#             plt.ylabel("IGD")
#             plt.yscale("log")
#         plt.legend()
#         problemfullname = '{}_obj{}'.format(problem_name, n_obj)
#         plt.savefig(os.path.join(maindir, "{}_iter_conv.png".format(problemfullname)))
#         plt.cla()
#         plt.clf()
