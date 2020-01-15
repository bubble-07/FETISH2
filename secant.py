import numpy as np
import scipy as sp
import regression
import idw
import nearestneighbor
import params

#Given design matrix X and outputs Y, computes linear
#cofficients for the best local linear fit about example i
#[this is in terms of yer usual column vec -> column vec transformer]
def get_linear_coefficients(X, Y, i):
    W = idw.get_idw_weight_matrix(X, i)
    coef_transform = regression.get_linear_coefficient_transform(X, W)
    coef_matrix = np.matmul(coef_transform, Y)
    return coef_matrix

#Given matrices X and Y, and a target output, compute/propose
#a new input vector to investigate which will get us close to the target
#uses local linear interpolation
def get_new_input_to_try_linear(X, Y, target):
    i = nearestneighbor.get_closest_index(Y, target)
    x = X[i]
    y = Y[i]
    X_shifted = X - x
    Y_shifted = Y - y
    A = get_linear_coefficients(X_shifted, Y_shifted, i)
    shifted_target = target - y
    v, _, _, _ = regression.linear_least_squares(np.transpose(A), shifted_target)
    v_norm = np.linalg.norm(v)
    print "Moving: ", v_norm
    r_scale = v_norm * params.SIGMA
    r = np.random.normal(loc=0.0, scale=r_scale, size=v.shape)

    return x + v + r

#Given a function f, and a pair of matrices (X, Y), perform one step of
#memory-secant optimization and return the new matrices (X', Y')
#which now have one more row
def memory_secant(f, mats, target, method=get_new_input_to_try_linear):
    X, Y = mats
    x = method(X, Y, target)
    y = f(x)
    return (np.vstack([X, x]), np.vstack([Y, y]))
