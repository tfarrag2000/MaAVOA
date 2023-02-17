import os

import numpy as np
import pickle

from pymoo.indicators.igd import IGD

dir = r'D:\My Research Results\Many_Objectives'

for probname in os.listdir(dir):
    probdir = os.path.join(dir, probname)
    if os.path.isdir(probdir):
        for runname in os.listdir(probdir):
            # print(probname,runname)
            if runname != "run_111" and  runname != "run_1111"  :
                continue


            rundir = os.path.join(probdir, runname)
            dir_pkl = os.path.join(rundir, 'result_object.pkl')
            if os.path.exists(dir_pkl):
                if os.path.exists(os.path.join(rundir, "history_igd_list.csv")):
                    print("{}_{}  done".format(probname, runname))
                    continue
                print("{}_{}  start".format(probname, runname))

                file = open(dir_pkl, 'rb')
                res=pickle.load(file)
                file.close()

                PF_filepath = os.path.join(rundir, "PF_new.csv")
                PF = np.genfromtxt(PF_filepath, delimiter=',')
                metric = IGD(PF, zero_to_one=True)
                F_list= [h.opt.get("F") for h in res.history]
                n_evals = np.array([e.evaluator.n_eval for e in res.history])
                igd_list = np.array([metric.do(_F) for _F in     F_list])
                np.savetxt(os.path.join(rundir, "history_igd_list.csv"), igd_list, delimiter=",")
                np.savetxt(os.path.join(rundir, "history_n_evals.csv"), n_evals, delimiter=",")

                # plt.plot(n_gen, igd_list, color='black', lw=0.7, label="Avg. CV of Pop")
                # # plt.scatter(runs, igd_list, facecolor="none", edgecolor='black', marker="p")
                # # plt.axhline(10 ** -2, color="red", label="10^-2", linestyle="--")
                # plt.title("Convergence")
                # plt.xlabel("no. of iterations")
