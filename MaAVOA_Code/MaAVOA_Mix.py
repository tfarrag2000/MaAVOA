import copy
import math
import random

import numpy as np
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.algorithm import Algorithm
from pymoo.core.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population, pop_from_array_or_individual
from pymoo.core.repair import NoRepair
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import has_feasible

from boundaryCheck import boundaryCheck
from exploitation import exploitation
from exploration import exploration


def comp_by_cv_then_random(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible just set random
        else:
            S[i] = np.random.choice([a, b])

    return S[:, None].astype(int)


class MaAVOA_Mix(Algorithm):

    def __init__(self,
                 ref_dirs,
                 pop_size=None,
                 sampling=FloatRandomSampling(),
                 n_offsprings=None,
                 eliminate_duplicates=DefaultDuplicateElimination(),
                 repair=None,
                 advance_after_initial_infill=True,
                 **kwargs
                 ):

        super().__init__(display=MultiObjectiveDisplay(), **kwargs)
        self.ref_dirs = ref_dirs

        if self.ref_dirs is not None:

            if pop_size is None:
                pop_size = len(self.ref_dirs)

            if pop_size < len(self.ref_dirs):
                print(
                    f"WARNING: pop_size={pop_size} is less than the number of reference directions ref_dirs={len(self.ref_dirs)}.\n"
                    "This might cause unwanted behavior of the algorithm. \n"
                    "Please make sure pop_size is equal or larger than the number of reference directions. ")

        if 'survival' in kwargs:
            survival = kwargs['survival']
            del kwargs['survival']
        else:
            survival = ReferenceDirectionSurvival(ref_dirs)

        if 'crossover' in kwargs:
            self.crossover = kwargs['crossover']
            del kwargs['crossover']
        if 'mutation' in kwargs:
            self.mutation = kwargs['mutation']
            del kwargs['mutation']

        # the population size used
        self.pop_size = pop_size

        # whether the algorithm should be advanced after initialization of not
        self.advance_after_initial_infill = advance_after_initial_infill

        # the survival for the genetic algorithm
        self.survival = survival

        # number of offsprings to generate through recombination
        self.n_offsprings = n_offsprings

        # if the number of offspring is not set - equal to population size
        if self.n_offsprings is None:
            self.n_offsprings = pop_size

        # set the duplicate detection class - a boolean value chooses the default duplicate detection
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()
        else:
            self.eliminate_duplicates = eliminate_duplicates

        # simply set the no repair object if it is None
        self.repair = repair if repair is not None else NoRepair()

        self.initialization = Initialization(sampling,
                                             repair=self.repair,
                                             eliminate_duplicates=self.eliminate_duplicates)

        # other run specific data updated whenever solve is called - to share them in all algorithms
        self.n_gen = None
        self.pop = None
        self.ARC = Population()
        self.MaAVOA_p1 = 0.7
        self.MaAVOA_p2 = 0.9

    def _setup(self, problem, **kwargs):

        if self.ref_dirs is not None:
            if self.ref_dirs.shape[1] != problem.n_obj:
                raise Exception(
                    "Dimensionality of reference points must be equal to the number of objectives: %s != %s" %
                    (self.ref_dirs.shape[1], problem.n_obj))

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.survival.opt

    def _initialize_infill(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        pop.set("n_gen", self.n_gen)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        if self.advance_after_initial_infill:
            self.pop = self.survival.do(self.problem, infills, n_survive=len(infills))
            pass

    def _infill(self):
        np.seterr(divide='ignore')

        # do the mating using the current population
        ########## create and update the archive
        ARC_off = self.__updateARC()
        ###################################

        Best_V1_X = self.__selectV1(self.ARC)
        Best_V2_X = self.__selectV2(Population.merge(self.pop, ARC_off))

        # pop_african,pop_unmutated= model_selection.train_test_split(self.pop,test_size=0.0, random_state=42)
        # ###############################
        # mutation=PolynomialMutation(eta=20, prob=None)
        # X_unmutated=pop_unmutated.get("X")
        # pop_mutated = mutation.do(self.problem, pop_unmutated)
        # X_mutated=pop_mutated.get("X")

        pop_african = self.pop
        ################### Africian exploration & exploitation ###################
        variables_no = self.problem.n_var
        X_african_all = pop_african.get("X")
        if self.MaAVOA_p1 != 1:
            indices = np.random.choice(X_african_all.shape[0], round(X_african_all.shape[0] * self.MaAVOA_p1),
                                       replace=False)
            X_african = X_african_all[indices]
        else:
            X_african = X_african_all

        p1 = 0.6
        p2 = 0.4
        p3 = 0.6
        alpha = 0.8
        betha = 0.2
        gamma = 2.5
        current_iter = self.n_gen
        max_iter = 500
        upper_bound = self.problem.xu[0]
        lower_bound = self.problem.xl[0]

        a = np.random.uniform(- 2, 2, (1, 1)) * ((np.sin((math.pi / 2) * (current_iter / max_iter)) ** gamma) + np.cos(
            (math.pi / 2) * (current_iter / max_iter)) - 1)
        P1 = (2 * np.random.rand() + 1) * (1 - (current_iter / max_iter)) + a

        # Update the location
        for i in range(X_african.shape[0]):
            current_V_X = X_african[i, :]
            F = P1 * (2 * np.random.rand() - 1)
            random_V_X = random.choice([Best_V1_X, Best_V2_X])
            if np.abs(F) >= 1:
                current_V_X = exploration(current_V_X, random_V_X, F, p1, upper_bound, lower_bound)
            else:
                if np.abs(F) < 1:
                    current_V_X = exploitation(current_V_X, Best_V1_X, Best_V2_X,
                                               random_V_X, F, p2, p3, variables_no, upper_bound,
                                               lower_bound)
            X_african[i, :] = current_V_X

        X_african_new = boundaryCheck(X_african, lower_bound, upper_bound)

        off_african = pop_from_array_or_individual(X_african_new)
        ##########################################################################
        mutation = PolynomialMutation(eta=20, prob=.2)
        off_mutated = mutation.do(self.problem, off_african)

        X_new = np.unique(np.concatenate((X_african_new.astype("float"), off_mutated.get("X").astype("float")), axis=0),
                          axis=0)

        off_new = pop_from_array_or_individual(X_new)
        # off_new =Population.merge(off_mutated,off_mutated)
        off = Population.merge(ARC_off, off_new)
        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(off) == 0:
            self.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        return off

    def __selectV1(self, ARC):

        if len(ARC) < 2:
            return ARC[0].get("X")

        V1_idx1, V1_idx2 = random.sample(range(len(ARC)), k=2)
        V1_1, V1_2 = ARC[V1_idx1], ARC[V1_idx2]

        if V1_1.data['niche'] < V1_2.data['niche']:
            return V1_1.get("X")
        else:
            return V1_2.get("X")

    def __selectV2(self, ARC):
        V2_Leaders = np.full((self.problem.n_obj,), None)
        min_F = np.full((self.problem.n_obj,), np.inf)

        for p in ARC:
            for i in range(len(min_F)):
                if p.F[i] < min_F[i]:
                    min_F[i] = p.F[i]
                    V2_Leaders[i] = p

        return random.choice(V2_Leaders).get("X")

    def __updateARC(self):
        FP_iter1 = self.survival.opt
        ARC_unsorted = Population.merge(self.ARC, FP_iter1)

        selection = TournamentSelection(func_comp=comp_by_cv_then_random)
        crossover = self.crossover  # SimulatedBinaryCrossover(eta=30, prob=1.0)
        mutation = self.mutation  # PolynomialMutation(eta=20, prob=None)

        n_select = math.ceil(len(ARC_unsorted) * self.MaAVOA_p2 / crossover.n_offsprings)
        parents = selection.do(ARC_unsorted, n_select, crossover.n_parents)
        ARC_off = crossover.do(self.problem, ARC_unsorted, parents)
        ARC_off = mutation.do(self.problem, ARC_off)
        ev = copy.deepcopy(self.evaluator)
        ev.eval(self.problem, ARC_off)

        ARC_unsorted = Population.merge(ARC_unsorted, ARC_off)
        survival1 = ReferenceDirectionSurvival(self.ref_dirs)
        xx = survival1.do(self.problem, ARC_unsorted, n_survive=100)  # 100 maximum number
        self.ARC = survival1.opt  # PF of the archive
        return ARC_off

    def _advance(self, infills=None, **kwargs):

        # merge the offsprings with the current population
        if infills is not None:
            self.pop = Population.merge(self.pop, infills)
        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(self.problem, self.pop, n_survive=self.pop_size, algorithm=self)
        pass
