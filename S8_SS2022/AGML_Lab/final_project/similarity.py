from enum import Enum
import numpy as np


class SimilarityMeasure(Enum):
    PAIRWISE_PEASON = 1
    GIVEN_MEAN_PEASON = 2


# mean is calculated once for every pairwise comparison (on the common rated items)
def pairwise_pearson(vector1, vector2):
    return given_mean_pearson(vector1, vector2, np.mean(vector1), np.mean(vector2))


# mean is calculated once for every user
def given_mean_pearson(vector1, vector2, mean1, mean2):
    centered1 = (vector1 - mean1)
    centered2 = (vector2 - mean2)
    # (unscaled)
    variance1 = np.sum(centered1**2)
    variance2 = np.sum(centered2**2)
    numerator = np.sum(centered1 * centered2)
    denominator = np.sqrt(variance1 * variance2)
    return numerator / denominator
