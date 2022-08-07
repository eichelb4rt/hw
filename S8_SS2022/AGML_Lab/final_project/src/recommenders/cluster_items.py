from __future__ import annotations
import numpy as np
from nptyping import NDArray

import config
import func.distance as distance
import func.similarity as similarity

from func.cluster import KMeans
from recommenders.item_based import REMOVED_ITEM, ItemBasedNeighborhoodRecommender


class ClusterItemsRecommender(ItemBasedNeighborhoodRecommender):
    name = "cluster_items"

    def __init__(self, k_neighbours, n_clusters, distance_measure=distance.rating_manhattan, epsilon=0.05, max_iterations=20, similarity_measure=similarity.pearson, min_similarity=None, weight_items=True, alpha=1, beta=None):
        self.kmeans = KMeans(n_clusters, distance_measure, epsilon, max_iterations)
        self.labels: NDArray[np.int8] = None
        super().__init__(k_neighbours, similarity_measure, min_similarity, weight_items, alpha, beta)

    def calc_similarities(self):
        """Calculate all the similarities between Items of a common cluster."""
        self.kmeans.fit(self.ratings_matix.T)
        self.labels = self.kmeans.label_all(self.ratings_matix.T)
        # return super().calc_similarities()
        similarities = np.full((self.n_items, self.n_items), -np.infty)
        # iterate through all classes
        for c in range(self.kmeans.k):
            # array of user indices
            class_items = np.where(self.labels == c)[0]
            # iterate through class items
            for idx, i in enumerate(class_items):
                for j in class_items[idx:]:
                    ij_similarity = self.single_similarity(i, j)
                    # similarity is symmetric
                    similarities[i, j] = ij_similarity
                    similarities[j, i] = ij_similarity
        return similarities

    def get_peers(self, user_idx, item_idx):
        # determine top k allowed items
        allowed_items = np.arange(self.n_items)
        # determine items who didn't rate or aren't similar enough - only in the same class this time
        same_class = self.labels == self.labels[item_idx]
        have_rated = self.ratings_matix[user_idx, :] != config.MISSING_RATING
        similar_enough = self.similarities[item_idx, :] > self.min_similarity if self.min_similarity is not None else True
        # removed = not (have rated and similar enough)
        removed_items = ~(have_rated & similar_enough & same_class)
        # mark the removed items with -1, apply order, remove marked items
        allowed_items[removed_items] = REMOVED_ITEM
        ordered_items = allowed_items[self.similarity_order[item_idx]]
        ordered_items = ordered_items[ordered_items != REMOVED_ITEM]
        # we use the top k of those ordered (by similarity) items (the user itself is already filtered out because his rating is missing)
        return ordered_items[-self.k_neighbours:]
