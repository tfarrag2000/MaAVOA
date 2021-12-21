# African Vulture Optimization alghorithm

# Read the following publication first and cite if you use it

# @article{abdollahzadeh2021african,
#   title={African Vultures Optimization Algorithm: A New Nature-Inspired Metaheuristic Algorithm for Global Optimization Problems},
#   author={Abdollahzadeh, Benyamin and Gharehchopogh, Farhad Soleimanian and Mirjalili, Seyedali},
#   journal={Computers \& Industrial Engineering},
#   pages={107408},
#   year={2021},
#   publisher={Elsevier},
#   url = {https://www.sciencedirect.com/science/article/pii/S0360835221003120}
# }

import matplotlib.pyplot as plt

# import matplotlib.ticker as ticker
# clear('all')
# close_('all')
# Population size and stoppoing condition
from AVOA import AVOA

pop_size = 200
max_iter = 10
# Define your objective function's details here
# fobj = ObjectiveFunction.ObjectiveFunction
# print(fobj)

variables_no = 4
lower_bound = 0
upper_bound = 1

Best_vulture1_F, Best_vulture1_X, convergence_curve = AVOA(pop_size, max_iter, lower_bound, upper_bound, variables_no)

# Best optimal values for the decision variables
plt.figure
plt.subplot(1, 2, 1)
plt.plot(Best_vulture1_X)
plt.xlabel('Decision variables')
plt.ylabel('Best estimated values ')
plt.box('on')
# Best convergence curve
plt.subplot(1, 2, 2)
plt.plot(convergence_curve)
plt.title('Convergence curve of AVOA')
plt.xlabel('Current_iteration')
plt.ylabel('Objective value')
plt.box('on')
plt.show()
