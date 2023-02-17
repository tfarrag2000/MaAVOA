import os

import numpy as np
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.nsga3 import NSGA3, ReferenceDirectionSurvival
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population, pop_from_array_or_individual
from pymoo.factory import get_termination, get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from EngProblems.EngProb1 import EngProb1

ALGORITHMS = [("MaAVOA_70_90",None), ("nsga3", NSGA3), ("unsga3", UNSGA3), #("moead", MOEAD),
                  ("ctaea", CTAEA)]
# termination = get_termination("n_eval", 100000) # run 2
termination = get_termination("n_gen", 500 ) # run 1
# termination = get_termination("time", "00:00:30")  # run 3
i=0
Total_X=None
for runId in range(1,16):
    for alg, algorithmClass in ALGORITHMS:
        problem_name = "EngProb1"
        maindir = r'D:\My Research Results\Many_Objectives'
        problemfullname = '{}_obj{}_{}'.format(problem_name, 4, alg)
        file = os.path.join(maindir, '{}\\run_{}\\X_new.csv'.format(problemfullname, runId))
        i = i + 1
        if os.path.exists(file):
            X = np.genfromtxt(file, delimiter=',')
            X=X.reshape((-1,10))
            if Total_X is None :
                Total_X=X
                continue
            Total_X=np.append(Total_X,X, axis=0)
            pass
# Total_X=np.unique(Total_X, axis=0)




n_obj=4
n_partitions = {3: (16, 0),4:(16, 0), 5: (6, 0), 8: (3, 2), 10: (3, 2), 15: (2, 1), 20: (2, 1), 30: (2, 1)}
ref_dirs = None
p1 = n_partitions[n_obj][0]
p2 = n_partitions[n_obj][1]
if p1 != 0:
    ref_dirs_L1 = get_reference_directions("das-dennis", n_dim=n_obj, n_partitions=p1)
    ref_dirs = ref_dirs_L1
if p2 != 0:
    ref_dirs_L2 = get_reference_directions("das-dennis", n_dim=n_obj, n_partitions=p2)
    ref_dirs = np.concatenate((ref_dirs, ref_dirs_L2))

p=EngProb1()
pop=pop_from_array_or_individual(Total_X)
Evaluator().eval(p,pop)
survival1 = ReferenceDirectionSurvival(ref_dirs)
xx = survival1.do(p, pop)  # 100 maximum number
PF = survival1.opt  # PF of the archive

PF_1=PF.get("F")
# np.savetxt('..\\PF\\PF_EngProb1_4.txt', PF_1, delimiter=',')
PX_1=PF.get("X")
# np.savetxt('..\\PF\\PF_X_EngProb1_4.txt', PX_1, delimiter=',')
print(len(PF_1))
#
# fronts, rank = NonDominatedSorting().do(Total_F, return_rank=True)
# non_dominated, last_front = fronts[0], fronts[-1]
pass