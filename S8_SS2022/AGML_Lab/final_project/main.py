import numpy as np
import config
import ratings
from recommender import ClusterUsersRecommender, ItemBasedNeighborhoodRecommender, MeanRecommender, RandomRecommender, UserBasedNeighborhoodRecommender, PredictionType


def main():
    # user, item
    X_QUALIFY = ratings.read("qualifying_blanc.csv")
    # user, item, rating
    X_TRAIN = ratings.read("train.csv")

    mean_recommender = MeanRecommender()
    ratings.fit_and_save(mean_recommender, X_TRAIN, X_QUALIFY)

    random_recommender = RandomRecommender(max_rating=config.MAX_RATING)
    ratings.fit_and_save(random_recommender, X_TRAIN, X_QUALIFY)

    # user_based_recommender = UserBasedNeighborhoodRecommender(k_neighbours=50, prediction_type=PredictionType.Z_SCORE, weight_items=True, min_similarity=0.4, beta=4)
    # ratings.fit_and_save(user_based_recommender, X_TRAIN, X_QUALIFY)
    
    # mean item similarity: 0.018
    # median item similarity: -0.023
    item_based_recommender = ItemBasedNeighborhoodRecommender(k_neighbours=50, weight_items=True, min_similarity=0.5, beta=6)
    ratings.fit_and_save(item_based_recommender, X_TRAIN, X_QUALIFY)

    # cluster_users_recommender = ClusterUsersRecommender(k_neighbours=50, n_clusters=8, prediction_type=PredictionType.Z_SCORE, weight_items=True, min_similarity=0.4, beta=4)
    # ratings.fit_and_save(cluster_users_recommender, X_TRAIN, X_QUALIFY)


if __name__ == "__main__":
    main()
