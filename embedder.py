import numpy as np
import scipy as sp
import random
from Schmear import *
from VectorSpace import *
from type_ids import *
from toposort import toposort_flatten
from interpreter_support import *
from rkhs import *
from rescaler_selection import *

#Responsible for maintaining information about all function
#embeddings based on some associated interpreter state
class EmbedderState(object):
    def __init__(self, interpreter_state):
        self.interpreter_state = interpreter_state

        #Dictionary from types to term-indexed arrays of means of embedding schmears
        self.means = {}

        #Dictionary from types to term-indexed arrays of covariance matrices of embedding schmears
        self.covariances = {}

        #Dictionary from types to space definitions
        self.spaces = {}

        #Dictionary from types to kernelized space embeddings of their terms
        self.kernel_spaces = {}

    def refresh(self):
        self.means = {}
        self.covariances = {}
        self.spaces = {}
        self.init_embeddings()

    def __str__(self):
        result = "Embeddings: \n"
        for kind in self.spaces:
            result += kind + ":\n"
            result += "\tSpace: \n" + str(self.spaces[kind])
            result += "\tMeans: \n" + str(self.means[kind])
            result += "\tCovariances: \n" + str(self.covariances[kind])
            result += "\n\n"
    
    def get_schmear_from_index(self, ind, kind):
        if isinstance(kind, VecType):
            space = self.interpreter_state.type_spaces[kind]
            vec = space.get(ind).get()
            return build_point_schmear(vec)

        space = self.spaces[kind][ind]
        mean = self.means[kind][ind]
        covariance = self.covariances[kind][ind]

        return Schmear(space, mean, covariance)

    def get_schmear_from_ptr(self, term_ptr):
        return self.get_schmear_from_index(term_ptr.get_index(), term_ptr.get_type())

    #Gets the full array of embedding means for a given kind
    def get_means_array(self, kind):
        if isinstance(kind, VecType): 
            space = self.interpreter_state.type_spaces[kind]
            result = []
            for ind in range(len(space.terms)):
                vec = space.get(ind).get()
                result.append(vec)
            return np.vstack(result)
        return self.means[kind]

    #Gets the full array of covariances for a given kind
    def get_covariances_array(self, kind):
        if isinstance(kind, VecType):
            space = self.interpreter_state.type_spaces[kind]
            num_terms = len(space.terms)
            return np.zeros([num_terms, kind.n, kind.n])
        return self.covariances[kind]

    #Gets the full array of inverse covariances for a given kind.
    #If the input is actually a vec type, then we actually kind
    #of don't do this, because it would be ill-defined -- instead,
    #we yield staccs of identity matrices
    def get_inv_covariances_array(self, kind):
        if isinstance(kind, VecType):
            space = self.interpreter_state.type_spaces[kind]
            num_terms = len(space.terms)
            eye = np.eye(kind.n)
            return np.tile(eye, (num_terms, 1, 1))
        return np.linalg.pinv(self.get_covariances_array(kind), hermitian=True)

    #Given a function row, return the kernelized input [mean] data matrix X, the
    #output [mean] data matrix Y and the output inverse-covariance tensor \Omega^{-1},
    def get_func_row_embeddings(self, arg_kernel_space, ret_type, func_row):
        arg_inds = []
        ret_inds = []
        for arg_ind in func_row:
            arg_inds.append(arg_ind)
            ret_inds.append(func_row[arg_ind])

        arg_inds = np.array(arg_inds)
        ret_inds = np.array(ret_inds)

        in_kernel_means = arg_kernel_space.K[arg_inds]

        out_means = get_means_array(ret_type)[ret_inds]
        out_inv_covar_mats = get_inv_covariances_array(ret_type)[ret_inds]

        return (in_kernel_means, out_means, out_inv_covar_mats)

    def update_embeddings(self):
        updated_means = {}
        updated_covariances = {}

        #To construct embeddings, we need to topologically sort
        #all types
        #Construct the dictionary of dependencies
        depends = {}
        for kind in self.interpreter_state.application_tables:
            depends[kind] = {kind.arg_type, kind.ret_type}

        #Contains types in dependency order
        sorted_types = toposort_flatten(depends, sort=False)

        for kind in sorted_types:
            if isinstance(kind, VecType):
                #Don't need to derive embeddings for vector types
                #but we should derive their spaces
                self.spaces[kind] = VectorSpace(kind.n)
                continue
            #Otherwise, must be a function

            #Derive the embedding for the type 
            arg_type = kind.arg_type
            ret_type = kind.ret_type
            space = self.interpreter_state.get_type_space(kind)
            table = self.interpreter_state.get_application_table(kind)

            #Get the argument kernel space if it already exists
            kernel_space = None
            if (arg_type in self.kernel_spaces):
                kernel_space = self.kernel_spaces[arg_type]
            else:
                #Otherwise, create it from scratch
                X = get_means_array(arg_type)
                scaler = get_rescaling(X)
                kernel_space = GaussianKernelSpace(X, scaler)
                self.kernel_spaces[arg_type] = kernel_space

            #Get the return space
            ret_space = self.spaces[ret_type]

            #build the function space
            func_space = KernelSumFunctionSpace(kernel_space, ret_space)
            self.spaces[kind] = func_space

            #Now actually compute the embedding for the function
            
            func_means = []
            func_covariances = []
            for func_ind in range(len(space.terms)):
                func_row = table.table[func_ind]
                #Obtain the tuple X_kernelized, Y, out_inv_covar_mats
                X_kernelized, Y, out_inv_covar_mats = self.get_func_row_embeddings(kernel_space, ret_type, func_row)
                #Obtain the prior mean and the prior covariance for the function
                prior_func_mean = self.get_means_array(kind)[func_ind]
                prior_func_covariance = self.get_covariances_array(kind)[func_ind]

                _, k = X_kernelized.shape 
                _, t = Y.shape

                prior_func_mean = np.reshape(prior_func_mean, (k, t))
                prior_func_covariance = np.reshape(prior_func_covariance, (k, t, k, t))

                #Do the regression
                post_func_mean, post_func_covariance = bayesian_multivariate_lin_reg(X_kernelized, Y,  out_inv_covar_mats, prior_func_mean, prior_func_covariance)

                #Flatten the func mean/covariance vecs
                k, t = post_func_mean.shape
                flat_mean = np.reshape(post_func_mean, k * t)
                flat_covariance = np.reshape(post_func_covariance, (k * t, k * t))

                func_means.append(flat_mean)
                func_covariances.append(flat_covariance)
            func_means = np.stack(func_means)
            func_covariances = np.stack(func_covariances)
            updated_means[kind] = func_means
            updated_covariances[kind] = func_covariances
        self.means = updated_means
        self.covariances = updated_covariances

    #Initialize mean, covariance matrices to all of their priors
    def init_embeddings(self):
        #To construct embeddings, we need to topologically sort
        #all types
        #Construct the dictionary of dependencies
        depends = {}
        for kind in self.interpreter_state.application_tables:
            depends[kind] = {kind.arg_type, kind.ret_type}

        #Contains types in dependency order
        sorted_types = toposort_flatten(depends, sort=False)

        for kind in sorted_types:
            if isinstance(kind, VecType):
                #Don't need to derive embeddings for vector types
                #But should give them spaces
                self.spaces[kind] = new VectorSpace(kind.n)
                continue
            #Otherwise, must be a function type
            arg_type = 


