import numpy as np
import config


def euclidean(vector_1, vector_2):
    common_dimensions = (vector_1 != config.MISSING_RATING) & (vector_2 != config.MISSING_RATING)
    n_dims = np.count_nonzero(common_dimensions)
    common_1 = vector_1[common_dimensions]
    common_2 = vector_2[common_dimensions]
    return np.linalg.norm(common_1 - common_2) / n_dims


def manhattan(vector_1, vector_2):
    common_dimensions = (vector_1 != config.MISSING_RATING) & (vector_2 != config.MISSING_RATING)
    n_dims = np.count_nonzero(common_dimensions)
    common_1 = vector_1[common_dimensions]
    common_2 = vector_2[common_dimensions]
    return np.sum(np.abs(common_1 - common_2)) / n_dims
