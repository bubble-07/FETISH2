import numpy as np
import scipy as sp

#Given an nxk design matrix [n samples, k dimensional inputs],
#an nxt output matrix Y [for t the output dimension], 
#a n x (t x t) array of output inverse-covariance matrices [one for each sample]
#, a prior mean [2D matrix with dims kxt], 
#and a prior covariance \Sigma_0 of dims
#kxt x kxt, return the posterior distribution over the model parameter B
#for the model Y=XB, in terms of its mean [a kxt matrix], and covariance
#[a kxt x kxt tensor]
def bayesian_multivariate_lin_reg(X, Y, out_inv_covar_mats, prior_mean, prior_covariance):
    n, k = X.shape
    _, t = Y.shape

    def kt_inverse(mat):
        mat_flat = np.reshape(mat, (k * t, k * t))
        mat_flat_inv = np.linalg.pinv(mat_flat, hermitian=True)
        return np.reshape(mat_flat_inv, (k, t, k, t))

    #/\_0 = \Sigma_0^-1
    prior_inverse_covariance = kt_inverse(prior_covariance)

    #Part I: Computing /\_n (shape (k x t) x (k x t))
    
    #Compute (I_t o X)^T \Omega^-1 (I_t o X), which is kxt x kxt
    A_contrib = np.einsum('nab,nc,nd->cadb', out_inv_covar_mats, X, X)

    #Compute /\_n = A_contrib + /\_0 
    A_n = A_contrib + prior_inverse_covariance



    #Part II: Computing u_n (shape k x t)

    #Compute (I_t o X)^T \Omega^-1, which is (n x t) x (k x t)
    #from things that are (n x t x t) and n x k
    u_contrib_pre = np.einsum('nab,nc->nacb', out_inv_covar_mats, X)

    #Compute (I_t o X)^T \Omega^-1 vec(Y), which is (k x t)
    u_contrib = np.einsum('ntab,nt->ab', u_contrib_pre, Y)

    #Compute /\_0 u_0, which is k x t
    u_prior = np.einsum('ktab,ab->kt', prior_inverse_covariance, prior_mean)

    #u_total is kxt
    u_total = u_contrib + u_prior

    #Compute /\_n^-1
    A_n_inv = kt_inverse(A_n)

    #u_n = /\_n^-1 * u_total
    u_n = np.einsum('ktab,ab->kt', A_n_inv, u_total)
    


    #Part III: Computing E[\sigma^2 | X, Y]
    #Reference: https://core.ac.uk/download/pdf/12171733.pdf 
    #Adapted for generalized linear regression, setting a(0) = 1

    #Compute residuals r = Y - X\hat{B}, of shape nxt
    r = Y - np.matmul(X, u_n)

    #Compute residual contrib r^T \Omega^-1 r (recall \Omega^-1 is of shape n x t x t)
    s_resid_contrib = np.einsum('nab,na,nb->')

    #Compute difference of means u_0 - u_n (k x t)
    mean_diff = prior_mean - u_n

    #Compute [/\_0^-1 + A_contrib^-1]^-1
    A_contrib_inv = kt_inverse(A_contrib)

    covariance_sum = prior_covariance + A_contrib_inv
    inv_covariance_sum = kt_inverse(covariance_sum)

    s_pdc_contrib = np.einsum('abcd,ab,cd->', inv_covariance_sum, mean_diff, mean_diff)

    s = (s_resid_contrib + s_pdc_contrib) / n


    #Part IV: Computing the total covariance = s * A_n_inv
    result_covariance = s * A_n_inv

    return (u_n, result_covariance)

