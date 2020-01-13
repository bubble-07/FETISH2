import numpy as np
import scipy as sp
import idw
import nearestneighbor

#Module for stuff we need to do to do coordinate descent on function
#directions

#Using IDW, given a D-dimensional input point and a collection
#of N function input (D-dimensional) , output pairs (K-dimensional), 
#return a NxK matrix of imputed outputs of each function on the given input
def impute_input_slice(point_in, func_pairs, vptrees, use_idw=False):
    samp_points = point_in.reshape((-1, 1))
    result = []

    if (use_idw == True):
        for input_points, output_points in func_pairs:
            output_on_samples = idw.idw_interpolate(samp_points, input_points, output_points)
            result.append(output_on_samples.reshape((-1)))
    else:
        samp_points = samp_points.reshape((1, -1))
        func_ind = 0
        for input_points, output_points in func_pairs:
            vptree = vptrees[func_ind]
            output_on_samples = nearestneighbor.interpolate(vptree, samp_points, input_points, output_points)
            result.append(output_on_samples.reshape((-1)))
            func_ind += 1
    return np.vstack(result)
