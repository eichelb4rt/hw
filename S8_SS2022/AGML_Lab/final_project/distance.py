import numpy as np


def euclidean(vector_1, vector_2):
    return np.linalg.norm(vector_1 - vector_2)


def manhattan(vector_1, vector_2):
    return np.sum(np.abs(vector_1 - vector_2))
