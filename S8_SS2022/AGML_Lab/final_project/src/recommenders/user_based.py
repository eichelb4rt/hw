from __future__ import annotations
import numpy as np
from enum import Enum
from nptyping import NDArray

import config
import func.clock as clock
import func.ratings as ratings
import func.similarity as similarity
from recommenders.recommender import Recommender


REGULAR_STANDARD_DEVIATION = 1
REMOVED_USER = -1


class PredictionType(Enum):
    RAW = 1
    CENTERED = 2
    Z_SCORE = 3


class UserBasedNeighborhoodRecommender(Recommender):
    """Recommends based on similar users.

    Parameters
    ----------
    k_neighbours: int
        Number of closest users being used for prediction.

    similarity_measure: (vectors, means, weights) => float
        A similarity measure for two users.

    prediction_type: PredictionType
        The algorithm used for prediction of ratings.

    min_similarity: Optional[float]
        Users with similarity < min_similarity are not used for prediction.

    pairwise_mean: bool
        True: Mean will be calculated pairwise on common items for prediction. False: Mean will be calculated once for every user.

    weight_items: bool
        True: Items will be weighted to counter long tail impact. False: All items have weight 1.

    alpha: float
        Similarity Amplifier: This is used to amplify the importance of similarity.

    beta: Optional[int]
        Discounted Similarity Threshold: If not None, this will be used to calculate the discounted similarity instead of the similarity.
    """

    name = "user_based"

    def __init__(self, k_neighbours, prediction_type=PredictionType.CENTERED, similarity_measure=similarity.pearson, min_similarity=None, pairwise_mean=True, weight_items=True, alpha=1, beta=None):
        self.k_neighbours = k_neighbours
        self.prediction_type = prediction_type
        self.similarity_measure = similarity_measure
        self.min_similarity = min_similarity
        self.pairwise_mean = pairwise_mean
        self.weight_items = weight_items
        self.alpha = alpha
        self.beta = beta

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
        # std dev of ratings of every user
        self.std_dev: NDArray[np.float32] = None
        # weights of items used for similarity
        self.item_weights: NDArray[np.float32] = None
        # similarities[u, v] = how similar are u and v?
        self.similarities: NDArray[np.float32] = None
        # rated_items[user, item] = has user rated item?
        self.rated_items: NDArray[bool] = None

        # similarity_order[u] = users orderered by similarity to u
        self.similarity_order: NDArray[np.int32] = None

    def fit(self, x_train) -> UserBasedNeighborhoodRecommender:
        self.users = ratings.user_map(x_train)
        self.items = ratings.item_map(x_train)
        self.n_users = len(self.users)
        self.n_items = len(self.items)

        self.ratings_matix = ratings.ratings_matrix(x_train, self.users, self.items)
        self.rated_items = self.ratings_matix != config.MISSING_RATING
        self.mean_ratings = np.mean(self.ratings_matix, axis=1, where=self.rated_items)
        if self.prediction_type == PredictionType.Z_SCORE:
            self.std_dev = self.calc_std_dev()
        if self.weight_items:
            self.item_weights = self.calc_item_weights()

        # cache similarities so we don't have to calculate it every time
        self.similarities = self.calc_similarities()

        self.similarity_order = np.argsort(self.similarities, axis=1)
        return self

    def rate(self, x_qualify):
        n_queries = len(x_qualify)
        predictions = np.empty(n_queries)
        for i, (user, item) in enumerate(x_qualify):
            predictions[i] = self.rate_single(user, item)
        # predictions can be slightly below 0 sometimes
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
        peer_group = self.get_peers(user_idx, item_idx)

        # if peer group is empty, just return mean rating of user
        if len(peer_group) == 0:
            return self.mean_ratings[user_idx]

        # calculate rating based on peer group
        peer_ratings = self.ratings_matix[peer_group, item_idx]
        peer_similarities = self.similarities[user_idx, peer_group]

        total_similarity = np.sum(np.abs(peer_similarities))
        if total_similarity == 0:
            return self.mean_ratings[user_idx]
        if self.prediction_type == PredictionType.RAW:
            weighted_ratings = peer_similarities * peer_ratings
            return np.sum(weighted_ratings) / total_similarity
        elif self.prediction_type == PredictionType.CENTERED:
            peer_means = self.mean_ratings[peer_group]
            weighted_ratings = peer_similarities * (peer_ratings - peer_means)
            return self.mean_ratings[user_idx] + np.sum(weighted_ratings) / total_similarity
        elif self.prediction_type == PredictionType.Z_SCORE:
            peer_means = self.mean_ratings[peer_group]
            peer_std_dev = self.std_dev[peer_group]
            weighted_ratings = peer_similarities * (peer_ratings - peer_means) / peer_std_dev
            return self.mean_ratings[user_idx] + self.std_dev[user_idx] * np.sum(weighted_ratings) / total_similarity

    def calc_similarities(self):
        """Calculate all the similarities between Users."""
        similarities = np.empty((self.n_users, self.n_users))
        for u in range(self.n_users):
            for v in range(u, self.n_users):
                uv_similarity = self.single_similarity(u, v)
                # similarity is symmetric
                similarities[u, v] = uv_similarity
                similarities[v, u] = uv_similarity
        return similarities

    def single_similarity(self, u, v) -> float:
        """Calculate the similarity between users u, v."""
        # items that were rated by both
        common_items = self.rated_items[u] & self.rated_items[v]
        # if there are no common items, 0 similarity
        n_common_items = np.count_nonzero(common_items)
        if n_common_items == 0:
            return 0
        # ratings for the common items
        u_ratings = self.ratings_matix[u][common_items]
        v_ratings = self.ratings_matix[v][common_items]
        # means used for similarity
        if self.pairwise_mean:
            u_mean = np.mean(u_ratings)
            v_mean = np.mean(v_ratings)
        else:
            u_mean = self.mean_ratings[u]
            v_mean = self.mean_ratings[v]
        # item weights used for similarity
        if self.weight_items:
            weights = self.item_weights[common_items]
        else:
            weights = np.ones(n_common_items)
        # similarity of common ratings
        uv_similarity = self.similarity_measure(u_ratings, v_ratings, u_mean, v_mean, weights)
        # discounted similarity if wanted
        if self.beta is not None:
            uv_similarity *= min(n_common_items, self.beta) / self.beta
        # amplify similarity
        if self.alpha != 1:
            uv_similarity = np.sign(uv_similarity) * np.abs(uv_similarity) ** self.alpha
        return uv_similarity

    def calc_std_dev(self):
        std_dev = np.empty(self.n_users)
        n_rated_items = np.count_nonzero(self.rated_items, axis=1)
        # std dev of items with 1 or less item are set to 0
        std_dev[n_rated_items <= 1] = 0
        # all others are computed
        to_be_computed = n_rated_items > 1
        mean_column = np.array([self.mean_ratings[to_be_computed]]).T
        centered = self.ratings_matix[to_be_computed, :] - mean_column
        # only include those items that have been rated
        have_been_rated = self.rated_items[to_be_computed, :]
        std_dev[to_be_computed] = np.sum(centered**2, axis=1, where=have_been_rated) / n_rated_items[to_be_computed]
        # standard value for std_dev as 1
        std_dev[std_dev == 0] = REGULAR_STANDARD_DEVIATION
        return std_dev

    def calc_item_weights(self):
        item_ratings = np.count_nonzero(self.rated_items, axis=0)
        return np.log(self.n_users / item_ratings)

    def get_peers(self, user_idx, item_idx):
        # determine top k allowed users
        allowed_users = np.arange(self.n_users)
        # determine users who didn't rate or aren't similar enough
        have_rated = self.ratings_matix[:, item_idx] != config.MISSING_RATING
        similar_enough = self.similarities[user_idx, :] > self.min_similarity if self.min_similarity is not None else True
        # removed = not (have rated and similar enough)
        removed_users = ~(have_rated & similar_enough)
        # mark the removed users with -1, apply order, remove marked users
        allowed_users[removed_users] = REMOVED_USER
        ordered_users = allowed_users[self.similarity_order[user_idx]]
        ordered_users = ordered_users[ordered_users != REMOVED_USER]
        # we use the top k of those ordered (by similarity) users (the user itself is already filtered out because his rating is missing)
        return ordered_users[-self.k_neighbours:]
