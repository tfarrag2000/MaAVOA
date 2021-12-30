from pymoo.core.individual import Individual


class empty_individual(Individual):

    def __init__(self, X=None, F=None, G=None,
                 dF=None, dG=None,
                 ddF=None, ddG=None,
                 CV=None, feasible=None,Cost=None,
                 **kwargs):
        super().__init__(X, F, G,
                         dF, dG,
                         ddF, ddG,
                         CV, feasible, **kwargs)
        self.Cost=Cost
        self.Rank = []
        self.DominationSet = []
        self.DominatedCount = None
        self.CrowdingDistance = None
