import numpy as np
import config


def euclidean(vector_1, vector_2):
    common_dimensions = (vector_1 != config.MISSING_RATING) & (vector_2 != config.MISSING_RATING)
    n_dims = np.count_nonzero(common_dimensions)
    # if no dimensions are common, assume maximum distance
    if n_dims == 0:
        return max_distance(euclidean, len(vector_1))
    common_1 = vector_1[common_dimensions]
    common_2 = vector_2[common_dimensions]
    return np.linalg.norm(common_1 - common_2) / n_dims


def manhattan(vector_1, vector_2):
    common_dimensions = (vector_1 != config.MISSING_RATING) & (vector_2 != config.MISSING_RATING)
    n_dims = np.count_nonzero(common_dimensions)
    if n_dims == 0:
        return max_distance(manhattan, len(vector_1))
    common_1 = vector_1[common_dimensions]
    common_2 = vector_2[common_dimensions]
    return np.sum(np.abs(common_1 - common_2)) / n_dims


def max_distance(distance_measure, n_dims):
    max_vector = np.full(n_dims, config.MAX_RATING)
    min_vector = np.full(n_dims, config.MIN_RATING)
    return distance_measure(max_vector, min_vector)
