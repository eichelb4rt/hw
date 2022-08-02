import numpy as np
import config
import ratings
from recommender import MeanRecommender, RandomRecommender, UserBasedNeighborhoodRecommender, PredictionType


def main():
    # user, item
    X_QUALIFY = ratings.read("qualifying_blanc.csv")
    # user, item, rating
    X_TRAIN = ratings.read("train.csv")

    mean_recommender = MeanRecommender()
    ratings.fit_and_save(mean_recommender, X_TRAIN, X_QUALIFY)

    random_recommender = RandomRecommender(max_rating=config.MAX_RATING)
    ratings.fit_and_save(random_recommender, X_TRAIN, X_QUALIFY)

    user_based_recommender = UserBasedNeighborhoodRecommender(k=10, prediction_type=PredictionType.Z_SCORE, min_similarity=-np.infty)
    ratings.fit_and_save(user_based_recommender, X_TRAIN, X_QUALIFY)


if __name__ == "__main__":
    main()
