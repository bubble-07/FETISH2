import numpy as np
import scipy as sp
import sklearn.random_projection

#Given a NxN projection matrix [columns are basis vecs]
#construct a DxN projection matrix where D <= N
def reduce_projection_dimension(X, dim_max):
    N, _ = X.shape
    D = min(N, dim_max)
    proj = sklearn.random_projection.GaussianRandomProjection(D)
    result_t = proj.fit_transform(np.transpose(X))
    return np.transpose(result_t)

