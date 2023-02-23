# import os
#
# import numpy as np
# import pickle
#
# from matplotlib import pyplot as plt
# from pymoo.indicators.igd import IGD
#
# dir = r'D:\My Research Results\Many_Objectives'
#
# for probname in os.listdir(dir):
#     probdir = os.path.join(dir, probname)
#     if os.path.isdir(probdir):
#         for runname in os.listdir(probdir):
#             # print(probname,runname)
#             # if runname != "run_1" and  runname != "run_11" :
#             #     continue
#
#             rundir = os.path.join(probdir, runname)
#             dir_pkl = os.path.join(rundir, 'result_object.pkl')
#             if os.path.exists(dir_pkl):
#                 if os.path.exists(os.path.join(rundir, "history_igd_list.csv")):
#                     print("{}_{}  done".format(probname, runname))
#                     continue
#                 print("{}_{}  start".format(probname, runname))
#
#                 file = open(dir_pkl, 'rb')
#                 res=pickle.load(file)
#                 file.close()
#
#                 PF_filepath = os.path.join(rundir, "PF_new.csv")
#                 PF = np.genfromtxt(PF_filepath, delimiter=',')
#
#
#                 metric = IGD(PF, zero_to_one=True)
#                 F_list= [h.opt.get("F") for h in res.history]
#                 n_evals = np.array([e.evaluator.n_eval for e in res.history])
#                 igd_list = np.array([metric.do(_F) for _F in     F_list])
#                 np.savetxt(os.path.join(rundir, "history_igd_list.csv"), igd_list, delimiter=",")
#                 np.savetxt(os.path.join(rundir, "history_n_evals.csv"), n_evals, delimiter=",")
#                 n_gen=range(1,len(igd_list)+1)
#
#                 # plt.plot(n_gen, igd_list, color='black', lw=0.7, label="Avg. CV of Pop")
#                 # # plt.scatter(runs, igd_list, facecolor="none", edgecolor='black', marker="p")
#                 # # plt.axhline(10 ** -2, color="red", label="10^-2", linestyle="--")
#                 # plt.title("Convergence")
#                 # plt.xlabel("no. of iterations")
#                 # plt.ylabel("IGD")
#                 # plt.yscale("log")
#                 # plt.legend()
#                 # plt.savefig(os.path.join(rundir, "conv.png"))
#                 # plt.cla()
#                 # plt.clf()
import os
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.algorithms.moo.age import AGEMOEA
import numpy as np
from matplotlib import pyplot as plt

ALGORITHMS = [("MaAVOA_70_90", "MaAVOA"), ("nsga3", "NSGA3"), ("unsga3", "UNSGA3"), ("moead", "MOEAD"),
              ("ctaea", "CTAEA"), ("AGEMOEA", "AGEMOEA")]
runID = "run_11"

for n_obj in [3, 5, 8, 10, 15]:
    for pID in [1, 2, 3, 4, 5, 6, 7]:
        for alg, Name in ALGORITHMS:
            problem_name = "dtlz{}".format(pID)
            maindir = r'C:\Many_Objectives\DTLZ_Problems'
            problemfullname = '{}_obj{}_{}'.format(problem_name, n_obj, alg)

            rundir = os.path.join(maindir, '{}\\{}'.format(problemfullname, runID))

            if not os.path.exists(os.path.join(rundir, "history_igd_list.csv")):
                continue
            print(problemfullname)
            igd_list = np.genfromtxt(os.path.join(rundir, "history_igd_list.csv"), delimiter=',')
            n_evals = np.genfromtxt(os.path.join(rundir, "history_n_evals.csv"), delimiter=',')
            n_gen = range(1, len(igd_list) + 1)

            plt.plot(n_gen, igd_list, lw=1, label=Name)
            # plt.scatter(runs, igd_list, facecolor="none", edgecolor='black', marker="p")
            # plt.axhline(10 ** -2, color="red", label="10^-2", linestyle="--")
            plt.title("Convergence")
            plt.xlabel("no. of iterations")
            plt.ylabel("IGD")
            plt.yscale("log")
        plt.legend()
        problemfullname = '{}_obj{}'.format(problem_name, n_obj)
        path = os.path.join(maindir, "{}_{}_iter_conv.png".format(problemfullname, runID))
        plt.savefig(path, dpi=1200)
        print("file saved : {}".format(path))
        plt.cla()
        plt.clf()
