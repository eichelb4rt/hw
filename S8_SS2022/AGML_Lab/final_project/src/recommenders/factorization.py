from __future__ import annotations
import numpy as np
from nptyping import NDArray
from sklearn.linear_model import Ridge

import config
import func.ratings as ratings
import func.distance as distance
from recommenders.recommender import Recommender


class ALS(Recommender):
    """Uses factorization to predict. (Learngin with Alternating Least Squares)
    """

    name = "als_factorization"

    def __init__(self, latent_dimensions=20, regularization_factor=5, epsilon=1e-1, max_iterations=10):
        self.latent_dimensions = latent_dimensions
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.ridge = Ridge(fit_intercept=False, alpha=regularization_factor)

    def fit(self, x_train) -> ALS:
        self.users = ratings.user_map(x_train)
        self.items = ratings.item_map(x_train)
        self.n_users = len(self.users)
        self.n_items = len(self.items)

        self.ratings_matix = ratings.ratings_matrix(x_train, self.users, self.items)
        self.rated = self.ratings_matix != config.MISSING_RATING

        # precompute biases
        self.total_mean = np.mean(self.ratings_matix, where=self.rated)
        self.biases = np.mean(self.ratings_matix, axis=1, where=self.rated) - self.total_mean
        self.intercepts = np.mean(self.ratings_matix, axis=0, where=self.rated) - self.total_mean

        # precompute targets
        self.p_targets = self.get_p_targets()
        self.q_targets = self.get_q_targets()

        # init factor vectors
        self.p = self.init_vectors(self.n_users)
        self.q = self.init_vectors(self.n_items)

        # start iterating
        iteration = 0
        change = np.infty
        while change > self.epsilon and iteration < self.max_iterations:
            old_p, old_q = self.p, self.q
            self.p = self.get_ps(self.q)
            self.q = self.get_qs(self.p)
            iteration += 1
            # total change is max distance of the new vectors to the old vectors
            p_change = np.max(distance.euclidean(self.p, old_p, axis=1))
            q_change = np.max(distance.euclidean(self.q, old_q, axis=1))
            change = p_change + q_change

        return self

    def init_vectors(self, n_vectors):
        """Produces random vectors with entries in [-1, 1)"""
        return 2 * np.random.rand(n_vectors, self.latent_dimensions) - 1

    def get_p_targets(self) -> NDArray[np.float32]:
        """The targets (y) is always the same in every iteration, so we can precompute them."""
        return [self.get_p_target(u) for u in range(self.n_users)]

    def get_q_targets(self) -> NDArray[np.float32]:
        """The targets (y) is always the same in every iteration, so we can precompute them."""
        return [self.get_q_target(i) for i in range(self.n_items)]

    def get_p_target(self, u) -> np.float32:
        # which ratings we even know
        known = self.rated[u, :]
        r_ui = self.ratings_matix[u, known]
        b_u = self.biases[u]
        b_i = self.intercepts[known]
        b_ui = b_u + b_i + self.total_mean
        # train ridge on ratings we know
        return r_ui - b_ui

    def get_q_target(self, i) -> np.float32:
        # which ratings we even know
        known = self.rated[:, i]
        r_ui = self.ratings_matix[known, i]
        b_u = self.biases[known]
        b_i = self.intercepts[i]
        b_ui = b_u + b_i + self.total_mean
        # train ridge on ratings we know
        return r_ui - b_ui

    def get_ps(self, qs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Gets p_i, the factor vectors for users."""
        return np.array([self.get_p(u, qs) for u in range(self.n_users)])

    def get_qs(self, ps: NDArray[np.float32]) -> NDArray[np.float32]:
        """Gets q_i, the factor vectors for items."""
        return np.array([self.get_q(i, ps) for i in range(self.n_items)])

    def get_p(self, u, qs: NDArray[np.float32]) -> NDArray[np.float32]:
        known = self.rated[u, :]
        x = qs[known]
        y = self.p_targets[u]
        self.ridge.fit(x, y)
        # theta is our p
        return self.ridge.coef_

    def get_q(self, i, ps: NDArray[np.float32]) -> NDArray[np.float32]:
        known = self.rated[:, i]
        x = ps[known]
        y = self.q_targets[i]
        self.ridge.fit(x, y)
        # theta is our q
        return self.ridge.coef_

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
            u = self.users[user]
        if not item_unknown:
            i = self.items[item]

        if user_unknown and item_unknown:
            return self.total_mean
        if user_unknown:
            return self.total_mean + self.intercepts[i]
        if item_unknown:
            return self.total_mean + self.biases[u]

        b_ui = self.biases[u] + self.intercepts[i] + self.total_mean
        return b_ui + self.p[u] @ self.q[i]


###############################################################################
################################ variable bias ################################
###############################################################################

class ALSVariableBiases(Recommender):
    """Uses factorization to predict. (Learning with Alternating Least Squares), but bias is not fixed here.
    """

    name = "als_bias"

    def __init__(self, latent_dimensions=20, regularization_factor=5, epsilon=1e-1, max_iterations=10):
        self.latent_dimensions = latent_dimensions
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.ridge = Ridge(fit_intercept=True, alpha=regularization_factor)

    def fit(self, x_train) -> ALSVariableBiases:
        self.users = ratings.user_map(x_train)
        self.items = ratings.item_map(x_train)
        self.n_users = len(self.users)
        self.n_items = len(self.items)

        self.ratings_matix = ratings.ratings_matrix(x_train, self.users, self.items)
        self.rated = self.ratings_matix != config.MISSING_RATING

        # precompute biases
        self.total_mean = np.mean(self.ratings_matix, where=self.rated)
        self.fixed_biases = np.mean(self.ratings_matix, axis=1, where=self.rated) - self.total_mean
        self.fixed_intercepts = np.mean(self.ratings_matix, axis=0, where=self.rated) - self.total_mean

        # init biases
        self.biases = np.full(self.n_users, 0, dtype=np.float32)
        self.intercepts = np.full(self.n_items, 0, dtype=np.float32)
        # init factor vectors
        self.p = self.init_vectors(self.n_users)
        self.q = self.init_vectors(self.n_items)

        # start iterating
        iteration = 0
        change = np.infty
        while change > self.epsilon and iteration < self.max_iterations:
            old_p, old_q = self.p, self.q
            self.p = self.get_ps(self.q)
            self.q = self.get_qs(self.p)
            iteration += 1
            # total change is max distance of the new vectors to the old vectors
            p_change = np.max(distance.euclidean(self.p, old_p, axis=1))
            q_change = np.max(distance.euclidean(self.q, old_q, axis=1))
            change = p_change + q_change

        return self

    def init_vectors(self, n_vectors):
        """Produces random vectors with entries in [-1, 1)"""
        return 2 * np.random.rand(n_vectors, self.latent_dimensions) - 1

    def get_p_target(self, u) -> np.float32:
        # which ratings we even know
        known = self.rated[u, :]
        r_ui = self.ratings_matix[u, known]
        b_i = self.intercepts[known]
        # train ridge on ratings we know
        return r_ui - (b_i + self.total_mean)

    def get_q_target(self, i) -> np.float32:
        # which ratings we even know
        known = self.rated[:, i]
        r_ui = self.ratings_matix[known, i]
        b_u = self.biases[known]
        # train ridge on ratings we know
        return r_ui - (b_u + self.total_mean)

    def get_ps(self, qs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Gets p_i, the factor vectors for users."""
        return np.array([self.get_p(u, qs) for u in range(self.n_users)])

    def get_qs(self, ps: NDArray[np.float32]) -> NDArray[np.float32]:
        """Gets q_i, the factor vectors for items."""
        return np.array([self.get_q(i, ps) for i in range(self.n_items)])

    def get_p(self, u, qs: NDArray[np.float32]) -> NDArray[np.float32]:
        known = self.rated[u, :]
        x = qs[known]
        y = self.get_p_target(u)
        self.ridge.fit(x, y)
        # update bias
        self.biases[u] = self.ridge.intercept_
        # theta is our p
        return self.ridge.coef_

    def get_q(self, i, ps: NDArray[np.float32]) -> NDArray[np.float32]:
        known = self.rated[:, i]
        x = ps[known]
        y = self.get_q_target(i)
        self.ridge.fit(x, y)
        # update intercept
        self.intercepts[i] = self.ridge.intercept_
        # theta is our q
        return self.ridge.coef_

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
            u = self.users[user]
        if not item_unknown:
            i = self.items[item]

        if user_unknown and item_unknown:
            return self.total_mean
        if user_unknown:
            return self.total_mean + self.intercepts[i]
        if item_unknown:
            return self.total_mean + self.biases[u]

        b_ui = self.biases[u] + self.intercepts[i] + self.total_mean
        return b_ui + self.p[u] @ self.q[i]
