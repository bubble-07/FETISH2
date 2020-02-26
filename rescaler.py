import numpy as np

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
