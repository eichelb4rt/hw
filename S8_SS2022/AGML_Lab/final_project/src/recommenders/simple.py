from __future__ import annotations
import numpy as np

from recommenders.recommender import Recommender


class MeanRecommender(Recommender):
    """Takes the mean of all recommendations.
    """

    name = "mean"

    def __init__(self):
        self.mean = 0

    def fit(self, x_train) -> MeanRecommender:
        self.mean = np.mean(x_train[:, 2])
        return self

    def rate(self, x_qualify):
        return np.full(x_qualify.shape[0], self.mean)


class RandomRecommender(Recommender):
    """Recommends completely random.

    Parameters
    ----------
    max_rating: int
        Ratings will be in range(max_rating).
    """

    name = "random"

    def __init__(self, max_rating=5):
        self.max_rating = max_rating

    def fit(self, x_train) -> RandomRecommender:
        return self

    def rate(self, x_qualify):
        return np.random.randint(self.max_rating, size=x_qualify.shape[0])
