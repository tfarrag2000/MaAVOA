import math
import random

import numpy as np
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.algorithm import Algorithm
from pymoo.core.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.core.initialization import Initialization
from pymoo.core.mating import Mating
from pymoo.core.population import Population, pop_from_array_or_individual
from pymoo.core.repair import NoRepair
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import has_feasible

from AVOA_ManyObjectives.boundaryCheck import boundaryCheck
from AVOA_ManyObjectives.exploitation import exploitation
from AVOA_ManyObjectives.exploration import exploration


class MaAVOA(Algorithm):

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

        super().__init__( display = MultiObjectiveDisplay(),**kwargs)
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

        # if mating is None:
        #     mating = Mating(selection,
        #                     crossover,
        #                     mutation,
        #                     repair=self.repair,
        #                     eliminate_duplicates=self.eliminate_duplicates,
        #                     n_max_iterations=100)
        # self.mating = mating

        # other run specific data updated whenever solve is called - to share them in all algorithms
        self.n_gen = None
        self.pop = None
        self.ARC=Population()

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

        # do the mating using the current population
        # off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)
        FP_iter =[]
        V2_Leaders = np.full((self.problem.n_obj,),None)
        Top_F=np.full((self.problem.n_obj,),np.inf)

        for p in self.pop:
            rank=p.data['rank']
            F=p.F
            if rank==0:
                FP_iter.append(p)

            for i in range(len(Top_F)):
                if F[i]<Top_F[i]:
                    Top_F[i]=F[i]
                    V2_Leaders[i]=p

        self.ARC =np.append(self.ARC ,FP_iter)


        Best_V1_X = self.__selectV1(self.ARC.tolist()).X
        Best_V2_X = self.__selectV2(V2_Leaders).X
        X=np.array([p.X for p in self.pop])
        variables_no=self.problem.n_var

        ################### Africian exploration & exploitation ###################
        p1 = 0.6
        p2 = 0.4
        p3 = 0.6
        alpha = 0.8
        betha = 0.2
        gamma = 2.5
        current_iter =1
        max_iter=1
        upper_bound = self.problem.xu[0]
        lower_bound =  self.problem.xl[0]

        a = np.random.uniform(- 2, 2, (1, 1)) * ((np.sin((math.pi / 2) * (current_iter / max_iter)) ** gamma) + np.cos(
            (math.pi / 2) * (current_iter / max_iter)) - 1)
        P1 = (2 * np.random.rand() + 1) * (1 - (current_iter / max_iter)) + a

        # Update the location
        for i in range(X.shape[0]):
            current_V_X = X[i, :]
            F = P1 * (2 * np.random.rand() - 1)
            random_V_X = random.choice([Best_V1_X, Best_V2_X])
            if np.abs(F) >= 1:
                current_V_X = exploration(current_V_X, random_V_X, F, p1, upper_bound, lower_bound)
            else:
                if np.abs(F) < 1:
                    current_V_X = exploitation(current_V_X, Best_V1_X, Best_V2_X,
                                               random_V_X, F, p2, p3, variables_no, upper_bound,
                                               lower_bound)
            X[i, :] = current_V_X

        X_new = boundaryCheck(X, lower_bound, upper_bound)

        off   =pop_from_array_or_individual(X_new)
        ##########################################################################
        mutation = PolynomialMutation(eta=20, prob=None)
        off=mutation.do(self.problem,off)
        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(off) == 0:
            self.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")

        return off

    def __selectV1(self, V1_Leaders):
        V1_Leaders = list(set(V1_Leaders))
        V1_1,V1_2= random.sample(V1_Leaders,k=2)

        if V1_1.data['niche']<V1_2.data['niche']:
            return V1_1
        else:
            return V1_2


    def __selectV2(self,V2_Leaders):

        return random.choice(V2_Leaders)

    def _advance(self, infills=None, **kwargs):

        # merge the offsprings with the current population
        if infills is not None:
            self.pop = Population.merge(self.pop, infills)
        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(self.problem, self.pop, n_survive=self.pop_size, algorithm=self)
        pass


