import numpy as np
import scipy as sp
import scipy.optimize
import matplotlib.pyplot as plt
import idw
import time
import glpk

glpk.env.term_on = False

#Given a data matrix X [dxn, columns are samples]
#a d-dimensional starting vector z and a d-dimensional
#direction vector [not necessarily normalized] v,
#compute the next iterate for the hit-and-run algorithm
def hit_and_run_iter(X, z, v):
    D, N = X.shape

    res_one = lin_prog_query(X, z, v)
    res_two = lin_prog_query(X, z, -v)

    #Interpret the optimization result, and return the next vector
    maximal_a = res_one[N]
    minimal_a = -res_two[N]

    picked_a = np.random.uniform(low=minimal_a, high=maximal_a)
    return z + v * picked_a

#Implements the method from
#https://arxiv.org/pdf/1402.4670.pdf
def overrelaxed_hit_and_run_iter(X, z, v):
    D, N = X.shape

    res_one = lin_prog_query(X, z, v)
    res_two = lin_prog_query(X, z, -v)

    #Interpret the optimization result, and return the next vector
    maximal_a = res_one[N]
    minimal_a = -res_two[N]

    L = maximal_a - minimal_a
    t_zero = -minimal_a

    A = 2.0 * (t_zero / L) - 1.0
    R = np.random.uniform(low=0.0, high=1.0)

    A_plus_one = 1.0 + A
    under_radical = A_plus_one * A_plus_one - 4.0 * A * R
    numerator = A_plus_one - np.sqrt(under_radical)
    t_one = L * (numerator / (2.0 * A))

    picked_a = minimal_a + t_one
    return z + v * picked_a


#Given a data matrix X [dxn, columns are samples]
#a d-dimensional starting vector z
#and the (dists, vecs) vector
#that one gets from "get_maximal_vertex_direction",
#perform one iteration of schmancy hit-and-run
def schmancy_hit_and_run_iter(X, z, dist_vec_pair):
    dists, vecs = dist_vec_pair
    D, N = X.shape
    X_relativized = X - z.reshape((D, 1))

    #The way we pick a direction is through rejection sampling
    #keep trying to pick until we get something
    while True:
        v = np.random.normal(size=D)
        v = v / np.linalg.norm(v)

        #project down the data matrix onto the hyperplane,
        #as this will be used to determine
        #proximity weights to each vertex
        X_proj = project_hyperplane(X_relativized, v)
        p = D - 1
        W = idw.get_idw_weights(np.transpose(X_proj))

        #Compute relativized estimated dists
        #for the candidate hyperplane
        #by measuring agreement of vecs with dists
        rel_dists = dists * np.abs(np.matmul(np.transpose(vecs), v))

        #Okay, now with the relativized estimated dists
        #in hand, compute the dist estimate using the weights
        est_dist = np.dot(W, rel_dists)

        max_dist = np.amax(rel_dists)

        r = est_dist / max_dist

        #Now, with probability r, accept the choice of v
        #otherwise, keep spinning.
        if (np.random.uniform() <= r):
            break
    return overrelaxed_hit_and_run_iter(X, z, v)


#Given a data matrix X [dxn, columns are samples],
#return a pair (dists, vecs)
#where dists is an array of n numbers, and vecs is a dxn array
#of unit vectors such that they are distances, directions
#of the paths to the furthest vertex from each vertex in X
def get_maximal_vertex_directions(X):
    X_T = np.transpose(X)
    dist_mat = sp.spatial.distance_matrix(X_T, X_T)
    max_dist_indices = np.argmax(dist_mat, axis=1)

    opp_vertices = X[:, max_dist_indices]
    unnorm_vecs = opp_vertices - X
    norms = np.linalg.norm(unnorm_vecs, axis=0, keepdims=True)
    vecs = unnorm_vecs / norms

    return (norms.reshape(-1), vecs)

#Given a data matrix X [dxn, columns are samples],
#project the data onto the plane normal to the unit vector
#v, and return the result
def project_hyperplane(X, v):
    #n-vector of projections
    projs = np.dot(np.transpose(X), v)
    sub = np.outer(v, projs)
    return X - sub

#Given a data matrix X [dxn, columns are samples],
#perform approximate normalization so that the convex hull
#most closely approximates a hypersphere
def covariance_matrix(X):
    return np.cov(X)

def get_centroid(P):
    D, N = P.shape
    return np.sum(P, axis=1) / N

#given
#Data matrix X [dxn, columns are samples]
#generate a uniform random convex combination of X's columns
def get_dirichlet_random(X):
    D, N = X.shape
    alphas = np.ones((N,))
    coefs = np.random.dirichlet(alphas)
    return np.matmul(X, coefs)

#Given a data matrix P [dxn, columns are samples],
#remove those columns which are convex combinations
#of the other columns to yield just the extreme points of the
#convex hull of the points.
#Algorithm adapted from https://link.springer.com/content/pdf/10.1007%2FBF02712874.pdf
def extrema(P):
    D, N = P.shape
    centroid = get_centroid(P)
    Q = np.zeros((D, 1))
    Q[:, 0] = centroid
    inds_to_process = set(range(N))
    while (len(inds_to_process) > 0):
        i = inds_to_process.pop()
        p = P[:, i]
        if (not convex_hull_query(Q, p)):
            #Perform a linear programming query from the centroid through p
            res = lin_prog_query(P, centroid, p - centroid)
            coefs = res[:N]
            nonzero_coef_inds = np.nonzero(coefs)[0]
            #Look now at only nonzero coefficients whose indices
            #are in inds_to_process
            for j in nonzero_coef_inds:
                if j in inds_to_process or j == i:
                    if (j != i):
                        inds_to_process.remove(j)
                    vertex = P[:, j].reshape((D, 1))
                    Q = np.hstack((Q, vertex))
    return Q[:, 1:]

#Query if z is in the convex hull of X [dxn, columns samples]
def convex_hull_query(X, z):
    #Done by solving
    #max 1
    #s.t: [[X]
    #     [1]] x = [z^T 1]
    #x[i] >= 0 \forall i
    D, N = X.shape

    lp = glpk.LPX()
    lp.obj.maximize = True
    lp.rows.add(D + 1)
    for i in range(D):
        lp.rows[i].bounds = z[i], z[i]
    lp.rows[D].bounds = 1.0, 1.0
    lp.cols.add(N)
    for i in range(N):
        lp.cols[i].bounds = 0.0, None
    lp.obj[:] = 0.0 * N

    coef_matrix = np.ones((D+1,N))
    coef_matrix[:D, :N] = X

    lp.matrix = np.reshape(coef_matrix, (-1))

    lp.simplex()

    return lp.status == 'opt' or lp.status == 'feas'

#Given a data matrix X [dxn, columns are samples]
#a d-dimensional starting vector z and a d-dimensional
#direction vector [not necessarily normalized] v,
#returns the vector of convex combination coefficients for the point within the
#convex hull of X which is furthest along v from z
#as the first N components, and the \alpha such that
#z + \alpha v is the found point as the last component
def lin_prog_query(X, z, v):
    #Done by solving     max a
    #s.t:
    #[[A -v]
    #[[1] 0]] [x^T a]^T=[z^T 1]^T
    #x[i] >= 0 \forall i

    D, N = X.shape

    lp = glpk.LPX()
    lp.obj.maximize = True
    lp.rows.add(D + 1)
    for i in range(D):
        lp.rows[i].bounds = z[i], z[i]
    lp.rows[D].bounds = 1.0, 1.0
    lp.cols.add(N + 1)
    for i in range(N + 1):
        lp.cols[i].bounds = 0.0, None
    lp.obj[:] = [0.0] * N + [1.0]

    coef_matrix = np.ones((D+1,N+1))
    coef_matrix[:D, :N] = X
    coef_matrix[D, N] = 0
    coef_matrix[:D, N] = -v

    lp.matrix = np.reshape(coef_matrix, (-1))

    lp.simplex()
    
    result = np.zeros(N + 1)
    for i in range(N + 1):
        result[i] = lp.cols[i].primal

    return result

def uniform_hit_and_run_step(X, z):
    D, N = X.shape

    v = np.random.normal(size=D)
    return hit_and_run_iter(X, z, v)

def schmancy_hit_and_run_a_while(X, n):
    D, _ = X.shape
    #Before doing _anything, pre-process X

    X = extrema(X)

    #Center about the centroid
    centroid = get_centroid(X).reshape((-1, 1))
    X_centered = X - centroid

    #Compute covariance matrix
    sigma = covariance_matrix(X_centered)

    #Invert covariance matrix
    try:
        sigma_inv = np.linalg.inv(sigma)
    except:
        #If not invertible, effectively ignore the unskewing step
        sigma = np.eye(D)
        sigma_inv = np.eye(D)

    X_unskewed = np.matmul(sigma_inv, X_centered)

    #From the unskewed X, now get the dist, vec maximal vertex directions
    dist_vec_pair = get_maximal_vertex_directions(X_unskewed)

    iters = []
    z = get_dirichlet_random(X)
    while len(iters) < n:
        z = schmancy_hit_and_run_iter(X, z, dist_vec_pair)
        iters.append(z)
    return np.array(iters)

def hit_and_run_a_while(X, n):
    iters = []
    z = get_dirichlet_random(X)
    while len(iters) < n:
        z = uniform_hit_and_run_step(X, z)
        iters.append(z)
    return np.array(iters)

'''
n_points = 10000
dim = 10
num_curves_to_average = 10

#test on an N-dimensional right simplex
X = np.eye(dim)
X[:, 0] *= 10.0
X = np.hstack((X, np.zeros((dim, 1))))

centroid_pos = get_centroid(X)
print centroid_pos

standard_curve = 0
schmancy_curve = 0

for i in range(num_curves_to_average):
    print "iteration", i

    start_time = time.time()
    standard_test = hit_and_run_a_while(X, n_points)
    elapsed = time.time() - start_time
    print "Standard method Elapsed time per iter (seconds): ", elapsed / n_points

    start_time = time.time()
    schmancy_test = schmancy_hit_and_run_a_while(X, n_points)
    elapsed = time.time() - start_time
    print "Shmancy method Elapsed time per iter (seconds): ", elapsed / n_points

    standard_test = standard_test.astype('float64')
    schmancy_test = schmancy_test.astype('float64')

    standard_diffs = standard_test - centroid_pos
    schmancy_diffs = schmancy_test - centroid_pos

    standard_cum_diffs = np.cumsum(standard_diffs, axis=0)
    schmancy_cum_diffs = np.cumsum(schmancy_diffs, axis=0)

    standard_cum_dists = np.linalg.norm(standard_cum_diffs, axis=1)
    schmancy_cum_dists = np.linalg.norm(schmancy_cum_diffs, axis=1)

    standard_dists = standard_cum_dists / (np.arange(n_points) + 1)
    schmancy_dists = schmancy_cum_dists / (np.arange(n_points) + 1)

    standard_curve += standard_dists / num_curves_to_average
    schmancy_curve += schmancy_dists / num_curves_to_average

plt.plot(np.arange(n_points), standard_curve, 'b')
plt.plot(np.arange(n_points), schmancy_curve, 'g')

plt.show()
'''
