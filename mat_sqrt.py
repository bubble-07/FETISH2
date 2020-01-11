import numpy as np

#Computes the square root of the "closest" positive semi-definite matrix
#to a given hermitian matrix
def sqrtm(X):
    D, W = np.linalg.eigh(X)
    D = np.maximum(0, D) #All must be non-negative due to the PSD assumption
    sqrt_D = np.sqrt(D)
    #The result comes from scaling the columns appropriately
    result = np.multiply(W, sqrt_D.reshape((-1, 1)))
    return result
