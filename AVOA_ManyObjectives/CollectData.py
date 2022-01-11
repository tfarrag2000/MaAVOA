import os

dir = r'D:\OneDrive\My Research\Many_Objectives\The Code\AVOA_ManyObjectives\results\NSGAIII_AfricanMutation'

with open(os.path.join(dir, 'summary_NSGAIII_AfricanMutation.csv'), 'w') as filesummary:
    filesummary.write("Fullname, probname,runname,n_gen, Obj_no ,pop_size ,igd, gd, exec_time\n")
    for probname in os.listdir(dir):
        probdir = os.path.join(dir, probname)
        if os.path.isdir(probdir):
            for runname in os.listdir(probdir):
                # print(probname,runname)
                rundir = os.path.join(probdir, runname)
                filepath = os.path.join(rundir, "final_result_run.csv")
                if os.path.exists(filepath):
                    with open(filepath, 'r') as file:
                        file.readline()  # title
                        data = file.readline()
                        prob = probname[10:11]
                        data = "{} ,{} ,{} ,{}".format(probname, probname[0:5], runname, data)
                        filesummary.write(data)
