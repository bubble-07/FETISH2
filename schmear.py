import numpy as np
import scipy as sp
from VectorSpace import *
from rkhs import *

#Class for "schmears" [distributions over some kind of space]
class Schmear(object):
    def __init__(self, space, mean, covariance_mat):
        self.space = space
        self.mean = mean
        self.covar_mat = covariance_mat
        self.covar_mat_inv = np.pinv(covariance_mat)

    #Combines two schmears by assuming that they're
    #direct contributions to the means and the covariances
    def combine_add(self, other):
        return Schmear(self.space, self.mean + other.mean, self.covar_mat + other.covar_mat)

    #Combines two schmears by performing a sensor-fusion
    #-like thing, where we're assuming a bayesian update
    #of this schmear w.r.t. the other
    def combine_fuse(self, other):
        result_covar_mat_inv = self.covar_mat_inv + other.covar_mat_inv
        result_covar_mat = np.pinv(result_covar_mat_inv)

        own_mean_weight = np.matmul(self.covar_mat_inv, self.mean)
        other_mean_weight = np.matmul(other.covar_mat_inv, other.mean)
        total_mean_weight = own_mean_weight + other_mean_weight

        result_mean = np.matmul(result_covar_mat, total_mean_weight)

        return Schmear(self.space, result_mean, result_covar_mat)
        
    def __str__(self):
        return "mean: " + str(self.mean) + " covariance: " + str(self.covar_mat)

def build_point_schmear(mean):
    k = mean.shape[0]
    return Schmear(VectorSpace(k), mean, np.zeros(k))

#Given a schmear in a function space and a schmear in its input space,
#return the imputed schmear on the output space
def impute_application_schmear(func_schmear, in_schmear, out_space):
    func_space = func_schmear.space
    func_mean = func_schmear.mean
    func_covar = func_schmear.covar_mat
    func = func_space.vec_to_func(func_mean)

    out_space = func_space.out_space

    in_space = in_schmear.space
    in_mean = in_schmear.mean
    in_covar = in_schmear.covar_mat

    out_zeroes = np.zeroes(func_space.out_dim)

    #Term 1: N(\bar f (\bar x), J_f_x, \Sigma_x J_f_x^T)

    #Compute the expected value func_mean(in_mean)
    out_mean = func.evaluate(in_mean)

    #Compute the covariance due to the input variance
    func_jacobian = func.get_jacobian(in_mean)
    out_covariance_from_input = np.matmul(func_jacobian, np.matmul(in_covar, np.transpose(func_jacobian)))

    out_schmear_from_input = Schmear(out_space, out_mean, out_covariance_from_input)

    #Term 2: sum_i e_i <N(\Sigma_f, K(x_{bar}, -)>, parallel component
    
    #First, compute \twiddle k
    kernel_space = func_space.space.get_kernel_space()
    K = kernel_space.K
    k_twiddle = kernel_space.kernelize(in_mean)

    K_inv = np.linalg.pinv(K, hermitian=True)
    #Compute k_twiddle^T K_inv k_twiddle = ||k_{||}||_H^2
    parallel_sq_norm = np.matmul(np.transpose(k_twiddle), np.matmul(K_inv, k_twiddle))
    #Compute K(in_mean, in_mean) (which is ||k||_H^2)
    total_sq_norm = kernel_space.scaling_factor
    #Subtract to find ||k_{_|_}||^2
    perp_sq_norm = total_sq_norm - parallel_sq_norm

    #Now, compute (I_t o k_twiddle^T) \Sigma_f (I_t o k_twiddle)

    #this is (txk)x(txk), and the k_twiddles are just k
    func_covar_tensor = func_space.mat_to_tensor(func_covar)
    out_covariance_from_func = np.einsum('satb,a,b->st', func_covar_tensor, k_twiddle, k_twiddle)

    out_schmear_from_func = Schmear(out_space, out_zeroes, out_covariance_from_func)

    #Term 3: Add in a term proportional to the squared magnitude of the perpendicular k-vec
    #and to (1 / alpha) * output_covariance

    out_prior_covariance = out_space.get_prior_covariance()
    unknown_scale_fac = perp_sq_norm / KERNEL_PRIOR_STRENGTH
    out_covariance_from_the_unknown = unknown_scale_fac * out_prior_covariance

    out_schmear_from_the_unknown = Schmear(out_space, out_zeroes, out_covariance_from_the_unknown)
    
    #Great, now add together all the schmears to yield the result
    input_plus_func = out_schmear_from_input.combine_add(out_schmear_from_func)
    all_together = input_plus_func.combine_add(out_schmear_from_the_unknown)
    return all_together
    


