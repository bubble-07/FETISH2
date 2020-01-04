import numpy as np
import sklearn.linear_model
from vptree import VpTree #From https://github.com/pderkowski/vptree

#Module for calculating the intrinsic dimension of a collection of points
#(one point per row)

THRESH_FOR_ESTIMATION = 8
KEEP_FRACTION=1.0

def estimate_intrinsic_dimension(X):
    N, D = X.shape
    upper_dim_bound = min(N, D)

    if (N < THRESH_FOR_ESTIMATION):
        return upper_dim_bound

    tree = VpTree(X)

    #Uses https://www.nature.com/articles/s41598-017-11873-y
    dists, _ = tree.getNearestNeighborsBatch(X, 3)
    dists = np.array(dists)
    closest_dists = dists[:, 1]
    second_closest_dists = dists[:, 2]

    mus = second_closest_dists / closest_dists
    mus = mus[closest_dists != 0]
    mus = mus[mus != 0]

    mus.sort()

    tot_mus = mus.shape[0]
    keep_mus = int(float(tot_mus) * KEEP_FRACTION)
    #Discard some mus
    mus = mus[:keep_mus]
    num_mus = mus.shape[0]
    Fs = (np.arange(num_mus) / float(num_mus)) + (0.0 / float(num_mus))
    log_mus = np.log(mus)
    log_Fs = -np.log(1.0 - Fs)

    log_mus = log_mus.reshape(-1, 1)
    log_Fs = log_Fs.reshape(-1, 1)

    reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
    reg.fit(log_mus, log_Fs)
    est = reg.coef_[0][0]
    print est

    return max(1, min(upper_dim_bound, est))
