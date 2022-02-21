import os

import numpy as np

dir = r'D:\My Research Results\Many_Objectives'

import os

with open(os.path.join(dir, 'summary.csv'), 'w') as filesummary:
    ALGORITHMS = [("MaAVOA_70_90", "MaAVOA"), ("nsga3", "NSGA3"), ("unsga3", "UNSGA3"), ("moead", "MOEAD"),("ctaea", "CTAEA")]
    header="problem, Obj_no, AlgorithmName, n_gen, n_eval, pop_size, exec_time,igd, gd,igd_plus, HV,igd, gd,igd_plus, HV"
    summaryheader = "runID,"
    for alg, Name in ALGORITHMS:
        summaryheader=summaryheader+Name+"_"+header+","
    summaryheader=summaryheader[:-1]+"\n"
    filesummary.write(summaryheader)

    for runsgroub in [[1, 11], [2, 22], [3, 32]]:
        for n_obj in [3,5,8,10,15]:
                for pID in [1, 2, 3, 4, 5, 6, 7]:
                    for runID in runsgroub:
                        fullrundata="run_{},".format(runID)
                        for alg, Name in ALGORITHMS:
                            problemfullname = 'dtlz{}_obj{}_{}'.format(pID, n_obj, alg)
                            probdir = os.path.join(dir, problemfullname)
                            rundir = os.path.join(probdir, "run_{}".format(runID))
                            resultfilepath= os.path.join(rundir, "final_result_run.csv")
                            # if runID > 3:
                            #     file =os.path.join(rundir, "result_object.pkl")
                            #     if os.path.exists(file):
                            #         os.remove(file)
                            # continue
                            if os.path.exists(resultfilepath):
                                with open(resultfilepath, 'r') as file:
                                    file.readline()  # title
                                    data = file.readline().strip('\n')
                                    m_filepath = os.path.join(rundir, "metrics.csv")
                                    if os.path.exists(m_filepath):
                                        with open(m_filepath, 'r') as f2:
                                            f2.readline()
                                            m=f2.readline().strip('\n')

                                    else:
                                        m="-1 ,-1 ,-1 ,-1"

                                    fullrundata = fullrundata+"{} ,{},".format(data,m)
                            else:
                                raise ("Error")
                        fullrundata = fullrundata[:-1] + "\n"
                        filesummary.write(fullrundata)
                    filesummary.write("summary\n")

#     for probname in os.listdir(dir):
#         probdir = os.path.join(dir, probname)
#         if os.path.isdir(probdir):
#             for runname in os.listdir(probdir):
#                 # print(probname,runname)
#                 # if runname != "run_2":
#                 #     continue
#                 rundir = os.path.join(probdir, runname)
#                 filepath = os.path.join(rundir, "final_result_run.csv")
#                 m_filepath = os.path.join(rundir, "metrics.csv")
#                 if os.path.exists(filepath):
#                     with open(filepath, 'r') as file:
#                         file.readline()  # title
#                         data = file.readline().strip('\n')
#                         if os.path.exists(m_filepath):
#                             with open(m_filepath, 'r') as f2:
#                                 f2.readline()
#                                 m=f2.readline().strip('\n')
#                         else:
#                             m="-1 ,-1 ,-1 ,-1"
#
#                         data = "{} ,{} ,{} ,{}\n".format(probname, runname, data,m)
#
#                         filesummary.write(data)
