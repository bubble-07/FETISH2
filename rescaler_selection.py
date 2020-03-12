import numpy as np
import math
import scipy as sp
import params
import rescaler

#Return the best rescaler of the data matrix X
def get_rescaling(X):
    N, K = X.shape
    if (N < 2):
        return rescaler.get_identity(K)
    if ((K < 2) or (not params.ENABLE_PCA_DESKEWING)):
        return rescaler.get_uniform_scaling(K, get_bandwidth_scaling(X))
    
    #Otherwise, we'll need to de-skew using PCA whitening
    centroid = get_centroid(X)
    X_centered = X - centroid

    #Get X = USW^T via SVD, whence X^T X = W S^2 W^T
    U, S, W_t = np.linalg.svd(X)

    S_inv = 1.0 / S

    #D is the decorrelation matrix now (for column vectors)
    D = S_inv * W_t

    #... but we want it on rows when it becomes a rescaler
    D_t = np.transpose(D)

    X_whitened = np.matmul(X_centered, M_t)
    #Compute the uniform scaling factor now that we've whitened the data
    scalefac = get_bandwidth_scaling(X_whitened)
    
    #The matrix we apply is the composite
    M = scalefac * D_t
    M_t = np.transpose(M)
    
    c = np.matmul(M_t, -centroid)

    return rescaler(M, c)

#Given a data matrix X, using the parameters
#KERNEL_BANDWIDTH_QUANTILE and KERNEL_BANDWIDTH_MULTIPLIER,
#return the uniform scaling factor to get the gaussian
#of unit stddev to the calculated bandwidth

def get_bandwidth_scaling(X):
    N, _ = X.shape
    if (N < 2):
        #Single data point -> no point in scaling
        return 1.0

    #Otherwise, use the specified quantile on the distance combined with a multiplier
    dist_mat = sp.spatial.distance.pdist(X, 'sqeuclidean') 
    mins = np.amin(dist_mat, axis=0) 
    epsilon = np.finfo(float).eps
    pre_mult_variance = np.quantile(mins, params.KERNEL_BANDWIDTH_QUANTILE)
    pre_mult_std_dev = math.sqrt(pre_mult_variance) + epsilon
    std_dev = params.KERNEL_BANDWIDTH_MULTIPLIER * pre_mult_std_dev
    return 1.0 / std_dev

def get_centroid(X):
    return np.mean(X, axis=1)


