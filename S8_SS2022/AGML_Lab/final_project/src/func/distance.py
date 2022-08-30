import numpy as np
import config


def euclidean(vector_1, vector_2, axis=0):
    return np.sum((vector_1 - vector_2)**2, axis=axis)


def manhattan(vector_1, vector_2, axis=0):
    return np.sum(np.abs(vector_1 - vector_2), axis=axis)

def rating_euclidean(rating_1, rating_2):
    return rating_distance(euclidean, rating_1, rating_2)

def rating_manhattan(rating_1, rating_2):
    return rating_distance(manhattan, rating_1, rating_2)


def rating_distance(distance_measure, rating1, rating2):
    common_dimensions = (rating1 != config.MISSING_RATING) & (rating2 != config.MISSING_RATING)
    n_dims = np.count_nonzero(common_dimensions)
    # if no dimensions are common, assume maximum distance
    if n_dims == 0:
        return max_distance(distance_measure, len(rating1))
    common_1 = rating1[common_dimensions]
    common_2 = rating2[common_dimensions]
    return distance_measure(common_1, common_2) / n_dims


def max_distance(distance_measure, n_dims):
    max_vector = np.full(n_dims, config.MAX_RATING)
    min_vector = np.full(n_dims, config.MIN_RATING)
    return distance_measure(max_vector, min_vector)
