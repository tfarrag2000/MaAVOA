import os

import numpy as np
from EngProblems.EngProb2 import EngProb2
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import pop_from_array_or_individual
from pymoo.factory import get_reference_directions

ALGORITHMS = [("MaAVOA_70_90", "MaAVOA"), ("nsga3", "NSGA3"), ("unsga3", "UNSGA3"), ("ctaea", "CTAEA"),
              ("AGEMOEA", "AGEMOEA")]
maindir = r'C:\Many_Objectives\EngProblem2'
p = EngProb2()
problem_name = "EngProb2"
i = 0
Total_X = None

for runId in [250, 500, 1000, 2000, 4000, 5000, 10000]:
    for alg, algorithmClass in ALGORITHMS:

        problemfullname = '{}_obj{}_{}'.format(problem_name, 4, alg)
        file = os.path.join(maindir, '{}\\run_{}\\X_new.csv'.format(problemfullname, runId))
        i = i + 1
        if os.path.exists(file):
            print(file)
            X = np.genfromtxt(file, delimiter=',')
            X = X.reshape((-1, 8))
            if Total_X is None:
                Total_X = X
                continue
            Total_X = np.append(Total_X, X, axis=0)
            pass
# Total_X=np.unique(Total_X, axis=0)


n_obj = 4
n_partitions = {3: (16, 0), 4: (16, 0), 5: (6, 0), 8: (3, 2), 10: (3, 2), 15: (2, 1), 20: (2, 1), 30: (2, 1)}
ref_dirs = None
p1 = n_partitions[n_obj][0]
p2 = n_partitions[n_obj][1]
if p1 != 0:
    ref_dirs_L1 = get_reference_directions("das-dennis", n_dim=n_obj, n_partitions=p1)
    ref_dirs = ref_dirs_L1
if p2 != 0:
    ref_dirs_L2 = get_reference_directions("das-dennis", n_dim=n_obj, n_partitions=p2)
    ref_dirs = np.concatenate((ref_dirs, ref_dirs_L2))

pop = pop_from_array_or_individual(Total_X)
Evaluator().eval(p, pop)
survival1 = ReferenceDirectionSurvival(ref_dirs)
xx = survival1.do(p, pop)  # 100 maximum number
PF = survival1.opt  # PF of the archive

PF_1 = PF.get("F")
np.savetxt('..\\PF\\PF_{}_4.txt'.format(problem_name), PF_1, delimiter=',')
PX_1 = PF.get("X")
np.savetxt('..\\PF\\PF_X_{}_4.txt'.format(problem_name), PX_1, delimiter=',')
print(len(PF_1))
#
# fronts, rank = NonDominatedSorting().do(Total_F, return_rank=True)
# non_dominated, last_front = fronts[0], fronts[-1]
pass
