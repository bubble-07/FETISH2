import numpy as np
import scipy as sp
import math
import params

#Represents a [1-dimensional] Gaussian RKHS which is
#given internally by a unit-variance kernel, but possibly
#with a rescaling first applied to the data matrix X
class GaussianKernelSpace(object):
    def __init__(self, X, rescaler):
        self.X = X
        self.rescaler = rescaler
        self.K = self.init_kernel_space(X, rescaler)
        self.K_inv = np.linalg.pinv(K)

    #Given a vector v which is in the same space as
    #the original X which was passed in, return
    #the vector corresponding to evaluating each kernel
    #function on v
    def kernelize(self, v):
        n, k = X.shape

        transformed_v = self.rescaler.transform_vec(v)  
        sq_dists = self.rescaled_X - transformed_v
        sq_dists = sq_dists * sq_dists
        sq_dists = np.sum(sq_dists, axis=1)

        exp_vec = np.exp(-0.5 * sq_dists)

        return self.scaling_factor * exp_vec

    #Given a vector v which is in the same space [r-dimensional] as
    #the original X which was passed in, return the k x r
    #jacobian of the kernelization operator at v
    def kernel_derivative(self, v):
        #v_k is k-dimensional, and in kernel space
        v_k = self.kernelize(v)  

        #compute v transformed [r-dimensional]
        transformed_v = self.rescaler.transform_vec(v)

        #Size: k x r
        diff = transformed_v - self.rescaled_X

        unscaled_result = v_k * diff

        result = np.matmul(unscaled_result, self.rescaler.jacobian())
        
        return result

    def init_kernel_space(self, X, rescaler):
        n, k = X.shape

        #Gaussian kernel: first, compute dot products between all X's
        self.rescaled_X = rescaler.transform(X)

        dist_mat = sp.spatial.distance.pdist(rescaled_X, 'sqeuclidean')

        #We pick a standard deviation of 1
        exp_mat = np.exp(-0.5 * dist_mat)

        self.scaling_factor = math.sqrt((1.0 / (2.0 * math.PI)) ** k)

        return self.scaling_factor * exp_mat

    def get_dimension(self):
        return self.K.shape[0]
    
    def get_embedding_of_index(self, ind):
        return self.K[ind]

#Representation of a space of functions whose
#domain is some space with the structure of a Gaussian
#kernel space on it, and whose codomain is some
#finite-dimensional vector space
class KernelSumFunctionSpace(object):
    def __init__(self, kernel_space, out_space):
        self.kernel_space = kernel_space
        self.out_space = out_space
        self.out_dim = out_space.get_dimension()
    def get_dimension(self):
        return self.kernel_space.get_dimension() * self.out_dim
    def get_kernel_space(self):
        return self.kernel_space
    def get_output_space(self):
        return self.out_space

    #Gets the prior covariance with dims (t * k) x (t * k) which
    #is given by \Sigma_Y o (1/(an) K^-1), where \Sigma_Y
    #is the txt covariance matrix for the prior on the output space
    def get_prior_covariance(self):
        k = self.kernel_space.get_dimension()
        t = self.out_dim
        n, _ = self.kernel_space.K.shape

        alpha = params.KERNEL_PRIOR_STRENGTH
        scale_factor = 1.0 / (n * alpha)
        scaled_K_inv = self.kernel_space.K_inv

        out_prior_covariance = self.out_space.get_prior_covariance()  

        kroned = np.kron(out_prior_covariance, scaled_K_inv) #TODO: is this the right ordering to work?
        return kroned

    #Same as above, but shape t x k x t x k
    def get_prior_covariance_tensor(self):
        k = self.kernel_space.get_dimension()
        t = self.out_dim

        kernel_tensor = get_kernel_tensor()
        return np.reshape(kernel_tensor, (t, k, t, k))
        
    def get_prior_mean(self):
        return np.zeros(self.get_dimension())

    #Convert from a KernelSumFunction to a 1d t * k vector
    def func_to_vec(self, func):
        return np.reshape(func.coef_matrix, -1)
    
    #Convert from a 1d t * k vector to a KernelSumFunction
    def vec_to_func(self, vec):
        k = self.kernel_space.get_dimension()
        t = self.out_dim
        coef_mat = np.reshape(vec, (t, k))
        return KernelSumFunction(self, coef_mat)
    
    #Convert from a (t * k) x (t * k) matrix to a (txk)x(txk) tensor
    def mat_to_tensor(self, mat):
        k = self.kernel_space.get_dimension()
        t = self.out_dim
        return np.reshape(mat, (t, k, t, k))

    #Convert from a (txk)x(txk) tensor to a (t * k) x (t * k) matrix
    def tensor_to_mat(self, tensor):
        k = self.kernel_space.get_dimension()
        t = self.out_dim
        return np.reshape(tensor, (t * k, t * k))

#Representation of a [multi-output] function
#which is expressible as a sum of kernel functions times basis elements
class KernelSumFunction(object):
    #coef_matrix is t x k, where k is the dimension of the kernel space
    #and t is the dimension of the output space. The original
    #dimension of the input space may be different [call this r]
    def __init__(self, space, coef_matrix):
        self.space = space
        self.coef_matrix = coef_matrix
        self.kernel_space = space.get_kernel_space

    #Given a vector in the original space of the input kernel space,
    #return the output from evaluating this function [a sum of kernel functions]
    #on that vector
    def evaluate(self, vec):
        k_vec = self.kernel_space.kernelize(vec)
        return np.matmul(self.coef_matrix, k_vec)

    #Obtain the jacobian in the original input space of this kernel sum function
    #with respect to the output space. size t x r.
    def get_jacobian(self, vec):
        #This is of size k x r
        kernel_jacobian = self.kernel_space.kernel_derivative(vec)


        return np.matmul(self.coef_matrix, kernel_jacobian)

        

