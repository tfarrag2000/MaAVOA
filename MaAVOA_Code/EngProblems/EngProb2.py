import numpy as np
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import random_by_bounds


class EngProb2(Problem):

    def __init__(self):
        self.m = 4
        self.name = "EngProb2"
        ff = (1 - (1 / np.power(10, 6)))
        xl = np.concatenate((np.full(self.m, 0.5, dtype=float), np.full(self.m, ff, dtype=float)), axis=0)
        xu = np.concatenate((np.full(self.m, 1.0, dtype=float), np.full(self.m, 10.0, dtype=float)), axis=0)

        super().__init__(n_var=2 * self.m, n_obj=4, xl=xl, xu=xu, n_constr=3)

    def _evaluate(self, X, out, *args, **kwargs):
        m = self.m
        F = []
        G = []
        for x in X:
            for i in range(m):
                x[i + m] = float(np.round(x[i + m]))

            f, g = self._evaluateSoultion(x)
            F.append(f)
            G.append(g)
        out["F"] = np.array(F)
        out["G"] = np.array(G)  # constraints
        pass

    def _evaluateSoultion(self, x):
        # print(x)
        np.seterr(divide='ignore')

        Fit = []
        f2 = 0
        f3 = 0
        f4 = 0
        R = []  # system reliability
        m = self.m  # number of subsystems
        # Calculating System reliability R for each subsystem, needed for f1
        for i in range(m):
            R.append(1 - np.power(1 - x[i], x[i + m]))

        # the following paprameters are given in table 2 in the engineering application paper
        alpha = [1, 2.3, 0.3, 2.3]  # given
        beta = [1.5, 1.5, 1.5, 1.5]  # given
        v = [1, 2, 3, 2]  # given
        w = [6, 6, 8, 7]  # given
        W = 500  # given
        V = 250
        C = 400  # given
        T = 1000  # given
        c = []

        f1 = 1  # Tamer
        for i in range(m):
            f1 = f1 * (1 - (np.power(1 - x[i], x[i + m])))
        # print("f1=", f1)
        Fit.append(f1 * -1)  ## -1 to convert max problem to min problem
        for i in range(m):
            f2 = f2 + (w[i] * np.power(v[i], 2) * np.power(x[i + m], 2))
        # print("f2=", f2)
        Fit.append(f2)
        # Calculating Cost for each subsystem, needed for f3
        for i in range(m):
            c.append(alpha[i] / 100000 * (np.power(-1 * T / np.log(x[i]), beta[i])))

        for i in range(m):
            f3 = f3 + (c[i] * (x[i + m] + np.exp(0.25 * x[i + m])))
        # print("f3=", f3)
        Fit.append(f3)

        min = w[0] * x[m] * np.exp(0.25 * x[m])
        for i in range(m):
            if i != 0:
                temp = (w[i] * x[i + m] * np.exp(0.25 * x[i + m]))
                if temp < min:
                    min = temp
        f4 = min
        # print("f4=", f4)
        Fit.append(f4)

        # constraints
        G = []
        G.append(f2 - V)
        G.append(f3 - C)
        G.append(f4 - W)

        return Fit, G
