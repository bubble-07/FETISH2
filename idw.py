import numpy as np
import scipy as sp
import traceback

#Given sample points [Nxd] and some known inputs [m x d] and outputs [m x d_two],
#return the [N x d_two] array which interpolates the value of a function
#at the sample points using inverse-distance weighting on the input space
def idw_interpolate(sample_points, known_inputs, known_outputs):
    N, d_in = sample_points.shape
    m, d_out = known_outputs.shape

    p = -(d_in + 1)

    #Shape is Nxm
    dist_mat = sp.spatial.distance_matrix(sample_points, known_inputs)

    #Pre-normalize the distance matrix to keep things in a good numerical range
    dist_mat = dist_mat / np.average(dist_mat)

    pow_dist_mat = np.power(dist_mat, p, where=dist_mat!=0)

    #Normalize each row
    idw_weight_mat = pow_dist_mat / np.sum(pow_dist_mat, axis=1, keepdims=True)
    #Set all zero-distance entries to 1.0
    idw_weight_mat[dist_mat==0] = 1.0
    #Normalize again
    idw_weight_mat = idw_weight_mat / np.sum(idw_weight_mat, axis=1, keepdims=True)

    return np.matmul(idw_weight_mat, known_outputs)

def normalize_weights(w):
    return w / np.linalg.norm(w, ord=1)

def get_idw_weights(X, i=None, p=None):
    N, M = X.shape
    if p is None:
        p = M + 1

    norms = np.linalg.norm(X, axis=1)
    max_norm = np.amax(norms)
    norms = norms / max_norm
    div_norms = np.power(norms, -p, where=norms!=0)

    if (max_norm == 0):
        traceback.print_stack()
        print X

    if i is None:
        div_norms[norms==0] = 0.0
    else:
        div_norms[i] = 0.0

    weights = normalize_weights(div_norms)    
    return weights

#Given the design matrix X whose rows are input-space
#examples and a particular example [row] index [about which X is centered]
#to interpolate around, compute the diagonal weighting matrix W which
#yields the proper inverse-distance-weighting coefficients
def get_idw_weight_matrix(X, i=None, p=None):
    return np.diag(get_idw_weights(X, i, p))

#Given a projecion matrix p_c [d, n] of input points, this computes
#the inverse distance weighting coefficients for imputation of each
#of the n points
#in the format [n, n], where each column sums to 1
def idw_coefficients(p_c):
    D, N = p_c.shape
    p_c_t = np.transpose(p_c)
    dists = sp.spatial.distance_matrix(p_c_t, p_c_t)
    #Compute the power to raise inverse distances to
    exponent = -N
    fracs = np.power(dists, exponent)
    #Normalize the rcolumns
    return fracs / fracs.sum(0)


