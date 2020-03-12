import numpy as np

def get_identity(k):
    return get_uniform_scaling(k, 1.0)

def get_uniform_scaling(k, scalefac):
    return np.ones(k) * scalefac

class AffineRescaler(object):
    def __init__(self, M, c):
        self.M = M
        self.M_t = np.transpose(M)
        self.M_inv = np.linalg.pinv(M)
        self.c = c

    #I _think_ this is right
    def jacobian(self):
        return self.M_t

    def transform_vec(self, v):
        return np.matmul(self.M_t, v) + c

    def transform(self, X):
        return np.matmul(X, self.M) + c

    def untransform(self, Y):
        return np.matmul(Y - c, self.M_inv)
