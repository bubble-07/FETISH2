import numpy as np
import params

class VectorSpace(object):
    def __init__(self, dim):
        self.dim = dim

    def get_dimension(self):
        return self.dim

    def get_prior_covariance(self):
        return params.VECTOR_PRIOR_STRENGTH * np.eye(self.dim)

    def get_prior_mean(self):
        return np.zeros(self.dim)
