from __future__ import annotations
import numpy as np
from nptyping import NDArray

import config
import func.distance as distance
import func.similarity as similarity

from func.cluster import KMeans
from recommenders.user_based import REMOVED_USER, UserBasedNeighborhoodRecommender, PredictionType


class ClusterUsersRecommender(UserBasedNeighborhoodRecommender):
    name = "cluster_users"

    def __init__(self, k_neighbours, n_clusters, distance_measure=distance.rating_manhattan, epsilon=0.05, max_iterations=20, prediction_type=PredictionType.CENTERED, similarity_measure=similarity.pearson, min_similarity=None, pairwise_mean=True, weight_items=True, alpha=1, beta=None):
        self.kmeans = KMeans(n_clusters, distance_measure, epsilon, max_iterations)
        self.labels: NDArray[np.int8] = None
        super().__init__(k_neighbours, prediction_type, similarity_measure, min_similarity, pairwise_mean, weight_items, alpha, beta)

    def calc_similarities(self):
        """Calculate all the similarities between Users of a common cluster."""
        self.kmeans.fit(self.ratings_matix)
        self.labels = self.kmeans.label_all(self.ratings_matix)
        # return super().calc_similarities()
        similarities = np.full((self.n_users, self.n_users), -np.infty)
        # iterate through all classes
        for c in range(self.kmeans.k):
            # array of user indices
            class_users = np.where(self.labels == c)[0]
            # iterate through class users
            for i, u in enumerate(class_users):
                for v in class_users[i:]:
                    uv_similarity = self.single_similarity(u, v)
                    # similarity is symmetric
                    similarities[u, v] = uv_similarity
                    similarities[v, u] = uv_similarity
        return similarities

    def get_peers(self, user_idx, item_idx):
        # determine top k allowed users
        allowed_users = np.arange(self.n_users)
        # determine users who didn't rate or aren't similar enough - only in the same class this time
        same_class = self.labels == self.labels[user_idx]
        have_rated = self.ratings_matix[:, item_idx] != config.MISSING_RATING
        similar_enough = self.similarities[user_idx, :] > self.min_similarity if self.min_similarity is not None else True
        # removed = not (have rated and similar enough)
        removed_users = ~(have_rated & similar_enough & same_class)
        # mark the removed users with -1, apply order, remove marked users
        allowed_users[removed_users] = REMOVED_USER
        ordered_users = allowed_users[self.similarity_order[user_idx]]
        ordered_users = ordered_users[ordered_users != REMOVED_USER]
        # we use the top k of those ordered (by similarity) users (the user itself is already filtered out because his rating is missing)
        return ordered_users[-self.k_neighbours:]
