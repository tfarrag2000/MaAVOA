import numpy as np

from pymoo.core.mutation import Mutation
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem


class PolynomialMutation(Mutation):
    def __init__(self, eta, prob=None):
        super().__init__()
        self.eta = float(eta)

        if prob is not None:
            self.prob = float(prob)
        else:
            self.prob = None

    def _do(self, problem, X, **kwargs):

        X = X.astype(float)
        Y = np.full(X.shape, np.inf)

        if self.prob is None:
            self.prob = 1.0 / problem.n_var



        do_mutation = np.random.random(X.shape) < self.prob

        a = np.random.uniform(- 2, 2, (1, 1)) * ((np.sin((math.pi / 2) * (current_iter / max_iter)) ** gamma) + np.cos(
            (math.pi / 2) * (current_iter / max_iter)) - 1)
        P1 = (2 * np.random.rand() + 1) * (1 - (current_iter / max_iter)) + a
        # Update the location
        for i in range(X.shape[0]):
            current_V_X = X[i, :]
            F = P1 * (2 * np.random.rand() - 1)
            random_V_X = random_select(current_V_X, Best_V1_X, Best_V2_X)
            if np.abs(F) >= 1:
                current_V_X = exploration(current_V_X, random_V_X, F, p1, upper_bound, lower_bound)
            else:
                if np.abs(F) < 1:
                    current_V_X = exploitation(current_V_X, Best_V1_X, Best_V2_X,
                                               random_V_X, F, p2, p3, variables_no, upper_bound,
                                               lower_bound)
            X[i, :] = current_V_X
        convergence_curve.append(Best_V1_individual.Cost[1])
        current_iter = current_iter + 1
        X_new = boundaryCheck(X, lower_bound, upper_bound)

        return Y


class PM(PolynomialMutation):
    pass