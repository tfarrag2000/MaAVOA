import os

import numpy as np

dir = r'D:\My Research Results\Many_Objectives'

with open(os.path.join(dir, 'summary.csv'), 'w') as filesummary:
    filesummary.write(
        "Fullname, runID, problem_Name, Obj_no, AlgorithmName, n_gen, n_eval, pop_size, exec_time,igd, gd,igd_plus, HV,igd, gd,igd_plus, HV\n")
    for probname in os.listdir(dir):
        probdir = os.path.join(dir, probname)
        if os.path.isdir(probdir):
            for runname in os.listdir(probdir):
                # print(probname,runname)
                # if runname != "run_2":
                #     continue
                rundir = os.path.join(probdir, runname)
                filepath = os.path.join(rundir, "final_result_run.csv")

                F_filepath = os.path.join(rundir, "PF.csv")
                F2_filepath = os.path.join(rundir, "PF_new.csv")
                m_filepath = os.path.join(rundir, "metrics.csv")

                if os.path.exists(F_filepath):
                    str = open(F_filepath, 'r').read()
                    str=str.replace(",\n", ",")
                    str = str.replace("[", "")
                    str = str.replace("]", "")
                    str = str.replace(" ", "")
                    open(F2_filepath, 'w').write(str)


                if os.path.exists(filepath):
                    with open(filepath, 'r') as file:
                        file.readline()  # title
                        data = file.readline().strip('\n')
                        if os.path.exists(m_filepath):
                            with open(m_filepath, 'r') as f2:
                                f2.readline()
                                m=f2.readline().strip('\n')
                        else:
                            m="-1 ,-1 ,-1 ,-1"

                        data = "{} ,{} ,{} ,{}\n".format(probname, runname, data,m)

                        filesummary.write(data)
