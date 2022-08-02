import numpy as np
import config
import ratings
from recommender import MeanRecommender, RandomRecommender, UserBasedNeighborhoodRecommender
import clock
import errors

ROTATIONS = 8


def main():
    # user, item, rating
    X = ratings.read("train.csv")
    np.random.shuffle(X)

    mean_recommender = MeanRecommender()
    clock.start(f"testing {mean_recommender.name}")
    mean_error = errors.cross_validate(mean_recommender, X, ROTATIONS)
    clock.stop(f"testing {mean_recommender.name}")
    print(f"{mean_recommender.name} error: {mean_error}")

    random_recommender = RandomRecommender(max_rating=config.MAX_RATING)
    clock.start(f"testing {random_recommender.name}")
    random_error = errors.cross_validate(random_recommender, X, ROTATIONS)
    clock.stop(f"testing {random_recommender.name}")
    print(f"{random_recommender.name} error: {random_error}")

    user_based_recommender = UserBasedNeighborhoodRecommender(k=10, min_similarity=-np.infty)
    clock.start(f"testing {user_based_recommender.name}")
    user_based_error = errors.cross_validate(user_based_recommender, X, ROTATIONS)
    clock.stop(f"testing {user_based_recommender.name}")
    print(f"{user_based_recommender.name} error: {user_based_error}")


if __name__ == "__main__":
    main()
