import os

dir = r'D:\OneDrive\My Research\Many_Objectives\The Code\AVOA_ManyObjectives\results'

with open(os.path.join(dir, 'summary.csv'), 'w') as filesummary:
    filesummary.write(
        "Fullname, runID, problem_Name, Obj_no, AlgorithmName, n_gen, n_eval, pop_size, exec_time,igd, gd, HV,igd_plus\n")
    for probname in os.listdir(dir):
        probdir = os.path.join(dir, probname)
        if os.path.isdir(probdir):
            for runname in os.listdir(probdir):
                # print(probname,runname)
                if runname != "run_2":
                    continue
                rundir = os.path.join(probdir, runname)
                filepath = os.path.join(rundir, "final_result_run.csv")
                if os.path.exists(filepath):
                    with open(filepath, 'r') as file:
                        file.readline()  # title
                        data = file.readline()
                        prob = probname[10:11]
                        data = "{} ,{} ,{} ".format(probname, runname, data)
                        filesummary.write(data)
