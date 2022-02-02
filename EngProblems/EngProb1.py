import numpy as np
from pymoo.core.problem import Problem


class EngProb1(Problem):

    def __init__(self, m=5):
        # # X1 = np.random.uniform(low=0, high=1, size=m)  # the reliability of a component ri
        # # X2 = np.random.randint(low=1, high=5, size=(m,))  # the number of components ni
        # X = []  # will be given not calculated here
        # for x in X1:
        #     X.append(x)
        # for x in X2:
        #     X.append(x)
        # print("X after first initialization:")
        # print("X=", X)
        self.m = m
        self.name = "EngProb1"
        xl = np.concatenate((np.zeros(m), np.zeros(m, dtype=int)), axis=0)
        xu = np.concatenate((np.ones(m), np.full(m, m, dtype=int)), axis=0)
        super().__init__(n_var=2 * m, n_obj=4, n_constr=3, xl=xl, xu=xu)

    # def _calc_pareto_front(self, n_pareto_points=100):
    #     x = anp.linspace(0, 1, n_pareto_points)
    #     return anp.array([x, 1 - anp.sqrt(x)]).T

    def _evaluate(self, X, out, *args, **kwargs):
        # print("we are in cs1")
        m = self.m
        F = []
        G = []
        for x in X:
            for i in range(m):
                x[i + m] = round(x[i + m])
            f, g = self._evaluateSoultion(x)
            F.append(f)
            G.append(g)
        out["F"] = np.array(F)
        out["G"] = np.array(G)  # constraints
        pass

    def _evaluateSoultion(self, x):
        Fit = []
        f2 = 0
        f3 = 0
        f4 = 0
        R = []  # system reliability
        m = self.m  # number of subsystems
        # Calculating System reliability R for each subsystem, needed for f1
        for i in range(m):
            # print("i=",i)
            R.append(1 - np.power(1 - x[i], x[i + m]))
        # print("R=", R)

        # the following paprameters are given in table 2 in the engineering application paper
        wv = [2, 4, 5, 8, 4]  # given and =w*power(v,2) in equation 6
        alpha = [2.5, 1.45, 0.541, 0.541, 2.1]  # given
        beta = [1.5, 1.5, 1.5, 1.5, 1.5]  # given
        w = [3.5, 4, 4, 3, 4.5]  # given
        V = [180, 180, 180, 180, 180]  # given
        C = [175, 175, 175, 175, 175]  # given
        W = [100, 100, 100, 100, 100]  # given

        f1 = 1 - (1 - R[0] * R[1]) * (1 - (R[2] + R[3] - (R[2] * R[3])) * R[4])
        # print("f1=", f1)
        Fit.append(f1)
        for i in range(m):
            f2 = f2 + (wv[i] * np.power(x[i + m], 2))
        # print("f2=", f2)
        Fit.append(f2)
        for i in range(m):
            f3 = f3 + (alpha[i] / 100000) * (np.power((-1000 / np.log(x[i])), beta[i])) * (
                    x[i + m] + np.exp(0.25 * x[i + m]))
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
        G.append(f2 - V[0])
        G.append(f3 - C[0])
        G.append(f4 - W[0])

        return Fit, G
