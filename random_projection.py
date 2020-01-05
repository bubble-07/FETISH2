import numpy as np
import scipy as sp
import sklearn.random_projection

#Given a NxN projection matrix [columns are basis vecs]
#construct a DxN projection matrix where D <= N
def reduce_projection_dimension(X, dim_max):
    N, _ = X.shape
    D = min(N, dim_max)
    
    rand_mat_t = np.random.normal(size=(N, D))
    Q, _ = np.linalg.qr(rand_mat_t)

    #Q is now of shape N, D
    Q_t = np.transpose(Q)

    return np.transpose(np.matmul(Q_t, X))


    '''
    sklearn's gaussian random projection is actually garbage,
    in that if N=D, norms are not preserved
    proj = sklearn.random_projection.GaussianRandomProjection(D)
    result_t = proj.fit_transform(np.transpose(X))
    return np.transpose(result_t)
    '''
