import numpy as np
import scipy as sp
import idw
import hit_n_run
import random_projection
import mat_sqrt

#TODO: Determine a good value for this hyperparameter
MAXIMAL_DIMENSION=30
MONTE_CARLO_ITERS=10

#Given a list of [embedded input points, embedded output points] function pairs
#and a [dxN] matrix X whose convex hull in the input we will compute dot products over,
#return the embedding matrix for ze functions
def get_embedding_matrix(func_pairs, X, iters=MONTE_CARLO_ITERS, dim_max=MAXIMAL_DIMENSION):
    dot_product_mat = get_dot_product_matrix(func_pairs, X, iters)

    #Extract matrix square root of the dot product matrix
    #That is, the dot product matrix is A^T A, but we want to recover A
    #If A^T A = U \Sigma V^T, then V \sqrt(\Sigma) V^T
    full_embedding_mat = mat_sqrt.sqrtm(dot_product_mat)

    #Project down to a smaller dimensionality
    reduced_embedding_mat = random_projection.reduce_projection_dimension(full_embedding_mat, dim_max) 

    return reduced_embedding_mat

#Given a list of (embedded input points, embedded output points) function pairs
#and a [dxN] matrix X whose convex hull in the input we will compute dot products
#over, return the dot-product matrix between ze functions
def get_dot_product_matrix(func_pairs, X, iters=MONTE_CARLO_ITERS):
    #Step 1: Use X to obtain a collection of sample points
    sample_points = hit_n_run.hit_and_run_a_while(X, iters)
    #sample_points is now [iters x d]

    outputs_on_samples = []

    #Step 2: Interpolate functions at the sample points
    for input_points, output_points in func_pairs:
        output_on_samples = idw.idw_interpolate(sample_points, input_points, output_points)
        outputs_on_samples.append(output_on_samples)

    #This is now num_funcs x num_samples x output_dim
    output_tensor = np.stack(outputs_on_samples)

    #Step 3: Compute the sum of gram matrices over all samples
    summed_gram_matrix = np.einsum('ijk,ljk->il', output_tensor, output_tensor)

    #Step 4: Normalize by the number of samples
    return summed_gram_matrix / iters

