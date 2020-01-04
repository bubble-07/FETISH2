import numpy as np
import scipy as sp
import idw

#Module for stuff we need to do to do coordinate descent on function
#directions

#Using IDW, given a D-dimensional input point and a collection
#of N function input (D-dimensional) , output pairs (K-dimensional), 
#return a NxK matrix of imputed outputs of each function on the given input
def impute_input_slice(point_in, func_pairs): 
    samp_points = point_in.reshape((-1, 1))
    result = []
    for input_points, output_points in func_pairs:
        output_on_samples = idw.idw_interpolate(samp_points, input_points, output_points)
        result.append(output_on_samples.reshape((-1)))
    return np.stack(result)
         

