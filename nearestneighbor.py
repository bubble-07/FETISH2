import numpy as np
import scipy as sp

#Given an output matrix Y [examples as rows], find the index
#of the row closest to the target
def get_closest_index(Y, target):
    subbed_Y = Y - target
    norms = np.linalg.norm(subbed_Y, axis=1)
    i = np.argmin(norms)
    return i
def interpolate(vptree, sample_points, input_points, output_points):
    _, batchIndices = vptree.getNearestNeighborsBatch(sample_points, 1)
    batchIndices = np.array(batchIndices)
    batchIndices = batchIndices.reshape(-1)
    return output_points[batchIndices]
