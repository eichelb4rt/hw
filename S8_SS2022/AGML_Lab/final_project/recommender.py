from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod

import distance


class Recommender(ABC):
    @abstractmethod
    def fit(self, x_train):
        pass

    @abstractmethod
    def rate(self, x_qualify) -> Recommender:
        pass


class MeanRecommender(Recommender):
    """Takes the mean of all recommendations.
    """

    def __init__(self):
        self.mean = 0

    def fit(self, x_train) -> MeanRecommender:
        self.mean = np.mean(x_train[:, 2])
        return self

    def rate(self, x_qualify):
        return np.full((x_qualify.shape[0], 1), self.mean)


class RandomRecommender(Recommender):
    """Recommends completely random.

    Parameters
    ----------
    max_rating: int
        Ratings will be in range(max_rating).
    """

    def __init__(self, max_rating=5):
        self.max_rating = max_rating

    def fit(self, x_train) -> RandomRecommender:
        return self

    def rate(self, x_qualify):
        return np.random.randint(self.max_rating, size=(x_qualify.shape[0], 1))


class SpecializedMeanRecommender(Recommender):
    """Recommends the mean of similar items. If it does not find similar items, recommend 0.

    Parameters
    ----------
    Recommender : _type_
        _description_
    """

    def __init__(self, distance_measure=distance.manhattan, radius=4):
        self.distance_measure = distance_measure
        self.radius = radius
        self.X_train = []

    def fit(self, x_train) -> SpecializedMeanRecommender:
        self.X_train = x_train
        return self

    def rate(self, x_qualify):
        recommendations = []
        for user, item in x_qualify:
            similar_items = [train_item for _, train_item, _ in self.X_train if self.similar(item, train_item)]
            # TODO: recommend mean of similar items
            # recommendations.append(np.mean)

    def similar(self, item_1, item_2):
        # TODO: how do i handle missing ratings?
        return True
