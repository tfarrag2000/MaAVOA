import numpy as np
import scipy
from scipy.spatial import distance


def calculateigd(truepf, pf):
    """Calculate IGD value for truepf and pf
        
    Parameters
    ----------
    truepf: "true" pf value 
    pf: estimated pf value
    
    Returns
    -------
    igd: IGD value
    
    """
    Y = scipy.spatial.distance.cdist(truepf, pf, 'euclidean')
    mindist = np.min(Y, axis=1)
    igd = np.mean(mindist)
    return igd


def IGD(truePF, PF):
    distEuclidean = []
    sumMin = 0
    for i in truePF:
        for x in PF:
            d = distance.euclidean(i, x)
            distEuclidean.append(d)
        distEuclidean.sort(reverse=True)
        sumMin += distEuclidean.pop()
        distEuclidean.clear()

    igdVal = sumMin / len(truePF)
    return igdVal
