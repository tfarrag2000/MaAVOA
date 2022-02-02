from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_reference_directions, get_termination
from pymoo.optimize import minimize

from EngProblems.EngProb1 import EngProb1
n_obj=4
# n_partitions = {3: (16, 0), 5: (6, 0), 8: (3, 2), 10: (3, 2), 15: (2, 1), 20: (2, 1), 30: (2, 1)}
# ref_dirs = None
# p1 = n_partitions[n_obj][0]
# p2 = n_partitions[n_obj][1]
# if p1 != 0:
#     ref_dirs_L1 = get_reference_directions("das-dennis", n_dim=n_obj, n_partitions=p1)
#     ref_dirs = ref_dirs_L1
# if p2 != 0:
#     ref_dirs_L2 = get_reference_directions("das-dennis", n_dim=n_obj, n_partitions=p2)
#     ref_dirs = np.concatenate((ref_dirs, ref_dirs_L2))

ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=12)

algorithm = NSGA3(ref_dirs=ref_dirs)
p=EngProb1()
termination = get_termination("n_gen", 10)

res = minimize(p,
               algorithm,
               termination,
               # seed=1,
               verbose=True,
               save_history=True)

# while result==None:
#       #intialization of X depends on the case, each case has different bounds
#       Xvector=case1.IntializeX("cs1")
#       result=p._evaluate(Xvector,"cs1")

pass
