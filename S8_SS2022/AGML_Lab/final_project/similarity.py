import numpy as np


def pearson(vector1, vector2, mean1, mean2, weights) -> float:
    centered1 = (vector1 - mean1)
    centered2 = (vector2 - mean2)
    # (unscaled)
    variance1 = np.sum(weights * centered1**2)
    variance2 = np.sum(weights * centered2**2)
    numerator = np.sum(weights * centered1 * centered2)
    denominator = np.sqrt(variance1 * variance2)
    if denominator == 0:
        return 0
    return numerator / denominator
