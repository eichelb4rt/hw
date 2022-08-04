import numpy as np
import config
import func.ratings as ratings
from recommenders.factorization import ALS
from recommenders.simple import MeanRecommender, RandomRecommender
from recommenders.user_based import UserBasedNeighborhoodRecommender
from recommenders.item_based import ItemBasedNeighborhoodRecommender
from recommenders.cluster_users import ClusterUsersRecommender
from recommenders.cluster_items import ClusterItemsRecommender


def main():
    # user, item
    X_QUALIFY = ratings.read("qualifying_blanc.csv")
    # user, item, rating
    X_TRAIN = ratings.read("train.csv")

    # mean_recommender = MeanRecommender()
    # ratings.fit_and_save(mean_recommender, X_TRAIN, X_QUALIFY)

    # random_recommender = RandomRecommender(max_rating=config.MAX_RATING)
    # ratings.fit_and_save(random_recommender, X_TRAIN, X_QUALIFY)

    # user_based_recommender = UserBasedNeighborhoodRecommender(k_neighbours=50, prediction_type=PredictionType.Z_SCORE, weight_items=True, min_similarity=0.4, beta=4)
    # ratings.fit_and_save(user_based_recommender, X_TRAIN, X_QUALIFY)
    
    # mean item similarity: 0.018
    # median item similarity: -0.023
    # item_based_recommender = ItemBasedNeighborhoodRecommender(k_neighbours=50, weight_items=True, min_similarity=0.5, beta=6)
    # ratings.fit_and_save(item_based_recommender, X_TRAIN, X_QUALIFY)

    # cluster_users_recommender = ClusterUsersRecommender(k_neighbours=50, n_clusters=8, prediction_type=PredictionType.Z_SCORE, weight_items=True, min_similarity=0.4, beta=4)
    # ratings.fit_and_save(cluster_users_recommender, X_TRAIN, X_QUALIFY)
    
    als_recommender = ALS(latent_dimensions=20, regularization_factor=5, epsilon=1e-1, max_iterations=10)
    ratings.fit_and_save(als_recommender, X_TRAIN, X_QUALIFY)


if __name__ == "__main__":
    main()
