import numpy as np
import scipy as sp

#Given a design matrix X and a weight matrix W,
#this yields the linear transform from
#[offsets of] output vectors
#to coefficient vectors [for a linear model]
def get_linear_coefficient_transform(X, W):
    XTW = np.matmul(np.transpose(X), W)
    XTWX = np.matmul(XTW, X)
    XTWX_inv = np.linalg.pinv(XTWX)
    return np.matmul(XTWX_inv, XTW)

#Given linear coefficients A and a vector b, find the x s.t. ||Ax-b|| is minimized
def linear_least_squares(A, b):
    return np.linalg.lstsq(A, b)
