import numpy as np
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import random_by_bounds


class EngProb1(Problem):

    def __init__(self):
        self.m = 5
        self.name = "EngProb1"
        xl = np.concatenate((np.zeros( self.m), np.full( self.m, 1, dtype=int)), axis=0)
        xu = np.concatenate((np.ones( self.m), np.full( self.m, self.m, dtype=int)), axis=0)
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
        wv = [2, 4, 5, 8, 4]  # given and =w*power(v,2) in equation 6
        alpha = [2.5, 1.45, 0.541, 0.541, 2.1]  # given
        beta = [1.5, 1.5, 1.5, 1.5, 1.5]  # given
        w = [3.5, 4, 4, 3, 4.5]  # given
        V = 180 # given
        C = 175  # given
        W = 100  # given

        f1 = (1 - (1 - R[0] * R[1]) * (1 - (R[2] + R[3] - (R[2] * R[3])) * R[4])) * -1
        # print("f1=", f1)
        Fit.append(f1)
        for i in range(m):
            f2 = f2 + (wv[i] * np.power(x[i + m], 2))

        # print("f2=", f2)
        Fit.append(f2)
        for i in range(m):
            f3 = f3 + (alpha[i] / 100000) * (np.power((-1000 / np.log(x[i])), beta[i])) * (x[i + m] + np.exp(0.25 * x[i + m]))
        # print("f3=", f3)
        Fit.append(f3)

        min = w[0] * x[m] * np.exp(0.25 * x[m])
        for i in range(m):
            if i != 0:
                temp = (w[i] * x[i + m] * np.exp(0.25 * x[i + m]))
                if temp < min:
                    min = temp
        f4 = min

        Fit.append(f4)

        # constraints
        G = []
        G.append(f2 - V)
        G.append(f3 - C)
        G.append(f4 - W)

        return Fit, G



