from __future__ import annotations
import numpy as np
from nptyping import NDArray

import config
import func.ratings as ratings
import func.similarity as similarity
from recommenders.recommender import Recommender


REMOVED_ITEM = -1


class ItemBasedNeighborhoodRecommender(Recommender):
    """Recommends based on similar items.

    Parameters
    ----------
    k_neighbours: int
        Number of closest items being used for prediction.

    similarity_measure: (vectors, means, weights) => float
        A similarity measure for two items.

    min_similarity: Optional[float]
        Items with similarity < min_similarity are not used for prediction.

    weight_items: bool
        True: Items will be weighted to counter long tail impact. False: All items have weight 1.

    alpha: float
        Similarity Amplifier: This is used to amplify the importance of similarity.

    beta: Optional[int]
        Discounted Similarity Threshold: If not None, this will be used to calculate the discounted similarity instead of the similarity.
    """

    name = "item_based"

    def __init__(self, k_neighbours, similarity_measure=similarity.pearson, min_similarity=None, weight_items=True, alpha=1, beta=None):
        self.k_neighbours = k_neighbours
        self.similarity_measure = similarity_measure
        self.min_similarity = min_similarity
        self.weight_users = weight_items
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
        # centered_ratings[u, i] = s_ui = r_ui - mu_u
        self.centered_ratings: NDArray[np.int8] = None
        # mean ratings of every user
        self.mean_ratings: NDArray[np.float32] = None
        # std dev of ratings of every user
        self.std_dev: NDArray[np.float32] = None
        # weights of items used for similarity
        self.user_weights: NDArray[np.float32] = None
        # similarities[u, v] = how similar are u and v?
        self.similarities: NDArray[np.float32] = None
        # rated_items[user, item] = has user rated item?
        self.rated_items: NDArray[bool] = None

        # similarity_order[u] = users orderered by similarity to u
        self.similarity_order: NDArray[np.int32] = None

    def fit(self, x_train) -> ItemBasedNeighborhoodRecommender:
        self.users = ratings.user_map(x_train)
        self.items = ratings.item_map(x_train)
        self.n_users = len(self.users)
        self.n_items = len(self.items)

        self.ratings_matix = ratings.ratings_matrix(x_train, self.users, self.items)
        self.rated_items = self.ratings_matix != config.MISSING_RATING
        self.mean_ratings = np.mean(self.ratings_matix, axis=1, where=self.rated_items)
        mean_column = np.array([self.mean_ratings]).T
        # center ratings
        self.centered_ratings = self.ratings_matix - mean_column
        # recover unrated (they get destroyed by centering)
        self.centered_ratings[~self.rated_items] = config.MISSING_RATING
        # weight users to combat long tail
        if self.weight_users:
            self.user_weights = self.calc_user_weights()

        # cache similarities so we don't have to calculate it every time
        self.similarities = self.calc_similarities()

        # similarity among items
        self.similarity_order = np.argsort(self.similarities, axis=0)
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

        # if peer group is empty, just return mean rating of item
        if len(peer_group) == 0:
            all_ratings = self.ratings_matix[:, item_idx]
            all_ratings = all_ratings[all_ratings != config.MISSING_RATING]
            return np.mean(all_ratings)

        # calculate rating based on peer group
        peer_ratings = self.ratings_matix[user_idx, peer_group]
        peer_similarities = self.similarities[peer_group, item_idx]

        total_similarity = np.sum(np.abs(peer_similarities))
        if total_similarity == 0:
            all_ratings = self.ratings_matix[:, item_idx]
            all_ratings = all_ratings[all_ratings != config.MISSING_RATING]
            return np.mean(all_ratings)

        weighted_ratings = peer_similarities * peer_ratings
        return np.sum(weighted_ratings) / total_similarity

    def calc_similarities(self):
        """Calculate all the similarities between Users."""
        similarities = np.empty((self.n_items, self.n_items))
        for i in range(self.n_items):
            for j in range(i, self.n_items):
                ij_similarity = self.single_similarity(i, j)
                # similarity is symmetric
                similarities[i, j] = ij_similarity
                similarities[j, i] = ij_similarity
        return similarities

    def single_similarity(self, i, j) -> float:
        """Calculate the similarity between users u, v."""
        # items that were rated by both
        common_users = self.rated_items[:, i] & self.rated_items[:, j]
        # if there are no common users, 0 similarity
        n_common_users = np.count_nonzero(common_users)
        if n_common_users == 0:
            return 0
        # ratings for the common users
        i_ratings = self.centered_ratings[:, i][common_users]
        j_ratings = self.centered_ratings[:, j][common_users]
        # item weights used for similarity
        if self.weight_users:
            weights = self.user_weights[common_users]
        else:
            weights = np.ones(n_common_users)
        # similarity of common ratings: using pearson this way results in the adjusted cosine
        ij_similarity = self.similarity_measure(i_ratings, j_ratings, 0, 0, weights)
        # discounted similarity if wanted
        if self.beta is not None:
            ij_similarity *= min(n_common_users, self.beta) / self.beta
        # amplify similarity
        if self.alpha != 1:
            ij_similarity = np.sign(ij_similarity) * np.abs(ij_similarity) ** self.alpha
        return ij_similarity

    def calc_user_weights(self):
        user_ratings = np.count_nonzero(self.rated_items, axis=1)
        return np.log(self.n_users / user_ratings)

    def get_peers(self, user_idx, item_idx):
        # determine top k allowed items
        allowed_items = np.arange(self.n_items)
        # determine items who didn't rate or aren't similar enough
        have_rated = self.ratings_matix[user_idx, :] != config.MISSING_RATING
        similar_enough = self.similarities[item_idx, :] > self.min_similarity if self.min_similarity is not None else True
        # removed = not (have rated and similar enough)
        removed_items = ~(have_rated & similar_enough)
        # mark the removed items with -1, apply order, remove marked items
        allowed_items[removed_items] = REMOVED_ITEM
        ordered_items = allowed_items[self.similarity_order[item_idx]]
        ordered_items = ordered_items[ordered_items != REMOVED_ITEM]
        # we use the top k of those ordered (by similarity) items (the user itself is already filtered out because his rating is missing)
        return ordered_items[-self.k_neighbours:]
