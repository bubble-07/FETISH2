import numpy as np

#Module for dimensionality reduction, in the sense that we're given
#a MxN matrix X and want to yield a transformation from that to some
#matrix which is RxN where R is the rank of X

#Given a MxN matrix, this returns a RxM matrix
#by which X may be left-multiplied to obtain a
#rank-R version of X
#This is done by computing the (reduced) SVD of X, U S V^T
#where U is MxK and V is KxN, and then finding
#the number of non-zero singular values (R)
#and truncating so that U is now MxR,
#S is RxR, and V is RxN
#Then we just return U^T
def get_rank_reduction_matrix(X):
    U, s, V = np.linalg.svd(X, full_matrices=False)
    eps = np.finfo(float).eps
    eps *= s.max() * max(X.shape) #Just following the value from np.linalg.matrix_rank
    R = (s >= eps).sum()
    reduced_U = U[:, :R]
    return np.transpose(reduced_U)
