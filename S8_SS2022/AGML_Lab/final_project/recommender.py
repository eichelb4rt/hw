from __future__ import annotations
from nptyping import NDArray
import numpy as np
from abc import ABC, abstractmethod

import distance
import ratings
import config
import similarity
from similarity import SimilarityMeasure


class Recommender(ABC):
    name: str

    @abstractmethod
    def fit(self, x_train):
        pass

    @abstractmethod
    def rate(self, x_qualify) -> Recommender:
        pass


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


class UserBasedNeighborhoodRecommender(Recommender):
    """Recommends the mean of similar items. If it does not find similar items, recommend 0.

    Parameters
    ----------
    k: int
        Number of closest users being used for prediction.

    min_similarity: Optional[float]
        Users with similarity < min_similarity are not used for prediction.
    """

    name = "user_based"

    def __init__(self, k, min_similarity=None, similarity_measure: SimilarityMeasure = SimilarityMeasure.PAIRWISE_PEASON):
        self.X_train = []
        self.k = k
        self.min_similarity = min_similarity
        self.similarity_measure = similarity_measure

        # user -> index
        self.n_users = 0
        self.users: dict[int, int] = {}

        # item -> index
        self.n_items = 0
        self.items: dict[int, int] = {}

        # ratings_matix[user, item] = rating
        self.ratings_matix: NDArray[np.int8] = None
        # mean ratings of every user
        self.mean_ratings: NDArray[np.float32] = None
        # similarities[u, v] = how similar are u and v?
        self.similarities: NDArray[np.float32] = None

        # similarity_order[u] = users orderered by similarity to u
        self.similarity_order: NDArray[np.int32] = None

    def fit(self, x_train) -> UserBasedNeighborhoodRecommender:
        self.X_train = x_train

        self.users = ratings.user_map(x_train)
        self.items = ratings.item_map(x_train)
        self.n_users = len(self.users)
        self.n_items = len(self.items)

        self.ratings_matix = ratings.ratings_matrix(x_train, self.users, self.items)
        self.mean_ratings = np.mean(self.ratings_matix, axis=1)

        # cache similarities so we don't have to calculate it every time
        self.similarities = np.empty((self.n_users, self.n_users))
        rated_items = self.ratings_matix != config.MISSING_RATING
        for u in range(self.n_users):
            for v in range(u, self.n_users):
                # items that were rated by both
                common_items = rated_items[u] * rated_items[v]
                # if there are no common items, - infinity similarity
                if np.count_nonzero(common_items) == 0:
                    uv_similarity = -np.infty
                    self.similarities[u, v] = uv_similarity
                    self.similarities[v, u] = uv_similarity
                    continue
                # ratings for the common items
                u_ratings = self.ratings_matix[u][common_items]
                v_ratings = self.ratings_matix[v][common_items]
                # similarity of common ratings
                # sorry, this was necessary because of the different runtime arguments of the different methods
                if self.similarity_measure == SimilarityMeasure.GIVEN_MEAN_PEASON:
                    uv_similarity = similarity.given_mean_pearson(u_ratings, v_ratings, self.mean_ratings[u], self.mean_ratings[v])
                elif self.similarity_measure == SimilarityMeasure.PAIRWISE_PEASON:
                    uv_similarity = similarity.pairwise_pearson(u_ratings, v_ratings)
                # similarity is symmetric
                self.similarities[u, v] = uv_similarity
                self.similarities[v, u] = uv_similarity

        self.similarity_order = np.argsort(self.similarities, axis=1)
        return self

    def rate(self, x_qualify):
        n_queries = len(x_qualify)
        predictions = np.empty(n_queries)
        for i, (user, item) in enumerate(x_qualify):
            predictions[i] = self.rate_single(user, item)
        return np.clip(predictions, config.MIN_RATING, config.MAX_RATING)

    def rate_single(self, user, item):
        user_unknown = user not in self.users
        item_unknown = item not in self.items

        if not user_unknown:
            user_idx = self.users[user]
        if not item_unknown:
            item_idx = self.items[item]

        # user didn't rate anything and item was never rated
        if user_unknown and item_unknown:
            return np.mean([config.MIN_RATING, config.MAX_RATING])

        # user didn't rate anything
        if user_unknown:
            all_ratings = self.ratings_matix[:, item_idx]
            all_ratings = all_ratings[all_ratings != config.MISSING_RATING]
            return np.mean(all_ratings)

        # item was never rated
        if item_unknown:
            all_ratings = self.ratings_matix[user_idx, :]
            all_ratings = all_ratings[all_ratings != config.MISSING_RATING]
            return np.mean(all_ratings)

        # if the user already rated this item, use that
        if self.ratings_matix[user_idx, item_idx] != config.MISSING_RATING:
            return self.ratings_matix[user_idx, item_idx]

        # user rated stuff and item was rated before, but user hasn't rated this one yet
        # -> use k most similar users, that have rated for this item, for prediction.

        # step 1: determine top k allowed users
        allowed_users = np.arange(self.n_users)
        # determine users who didn't rate or aren't similar enough
        have_rated = self.ratings_matix[:, item_idx] != config.MISSING_RATING
        similar_enough = self.similarities[user_idx, :] > self.min_similarity if self.min_similarity is not None else True
        removed_users = 1 - have_rated * similar_enough
        # mark the removed users with -1, apply order, remove marked users
        allowed_users[removed_users] = -1
        ordered_users = allowed_users[self.similarity_order[user_idx]]
        ordered_users = ordered_users[ordered_users != -1]
        # we use the top k of those ordered (by similarity) users (the user itself is already filtered out because his rating is missing)
        peer_group = ordered_users[-self.k:]
        # if peer group is empty, just return mean rating of user
        if len(peer_group == 0):
            return self.mean_ratings[user_idx]

        # step 2: calculate rating based on peer group
        all_ratings = self.ratings_matix[:, item_idx]
        all_similarities = self.similarities[user_idx, :]

        peer_ratings = all_ratings[peer_group]
        peer_means = self.mean_ratings[peer_group]
        peer_similarities = all_similarities[peer_group]

        total_similarity = np.sum(np.abs(peer_similarities))
        weighted_ratings = peer_similarities * (peer_ratings - peer_means)
        return self.mean_ratings[user_idx] + np.sum(weighted_ratings) / total_similarity
