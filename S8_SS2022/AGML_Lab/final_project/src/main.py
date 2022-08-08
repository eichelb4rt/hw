import numpy as np

import config
import func.ratings as ratings
import func.distance as distance

from recommenders.factorization import ALS, ALSVariableBiases
from recommenders.hybrid import Hybrid, MinimizationGoal
from recommenders.simple import MeanRecommender, RandomRecommender
from recommenders.user_based import PredictionType, UserBasedNeighborhoodRecommender
from recommenders.item_based import ItemBasedNeighborhoodRecommender
from recommenders.cluster_users import ClusterUsersRecommender
from recommenders.cluster_items import ClusterItemsRecommender


ROUND_PREDICTIONS = True


def main():
    # user, item
    X_QUALIFY = ratings.read("qualifying_blanc.csv")
    # user, item, rating
    X_TRAIN = ratings.read("train.csv")

    # mean_recommender = MeanRecommender()
    # ratings.fit_and_save(mean_recommender, X_TRAIN, X_QUALIFY, ROUND_PREDICTIONS)

    # random_recommender = RandomRecommender(max_rating=config.MAX_RATING)
    # ratings.fit_and_save(random_recommender, X_TRAIN, X_QUALIFY, ROUND_PREDICTIONS)

    user_based_recommender = UserBasedNeighborhoodRecommender(k_neighbours=50, prediction_type=PredictionType.Z_SCORE, weight_items=True, min_similarity=0.4, beta=4)
    # ratings.fit_and_save(user_based_recommender, X_TRAIN, X_QUALIFY, ROUND_PREDICTIONS)

    item_based_recommender = ItemBasedNeighborhoodRecommender(k_neighbours=50, weight_items=True, min_similarity=0.5, beta=6)
    ratings.fit_and_save(item_based_recommender, X_TRAIN, X_QUALIFY, ROUND_PREDICTIONS)

    cluster_users_recommender = ClusterUsersRecommender(k_neighbours=50, n_clusters=8, prediction_type=PredictionType.Z_SCORE, weight_items=True, min_similarity=0.4, beta=4)
    ratings.fit_and_save(cluster_users_recommender, X_TRAIN, X_QUALIFY, ROUND_PREDICTIONS)
    
    cluster_items_recommender = ClusterItemsRecommender(k_neighbours=50, n_clusters=8, weight_items=True, min_similarity=0.4, beta=4)

    als_recommender = ALS(latent_dimensions=20, regularization_factor=5, epsilon=1e-1, max_iterations=10)
    # ratings.fit_and_save(als_recommender, X_TRAIN, X_QUALIFY, ROUND_PREDICTIONS)
    
    als_variable_bias = ALSVariableBiases(latent_dimensions=20, regularization_factor=5, epsilon=1e-2, max_iterations=20)
    ratings.fit_and_save(als_variable_bias, X_TRAIN, X_QUALIFY, ROUND_PREDICTIONS)
    
    hybrid_recommender = Hybrid([cluster_users_recommender, item_based_recommender, als_variable_bias], min_goal=MinimizationGoal.MEAN_SQUARED_ERROR, epsilon=1e-4, lr=1, max_iterations=100, plot_descent=True)
    ratings.fit_and_save(hybrid_recommender, X_TRAIN, X_QUALIFY, ROUND_PREDICTIONS)


if __name__ == "__main__":
    main()
