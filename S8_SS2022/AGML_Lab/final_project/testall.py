import numpy as np
import config
import ratings
from recommender import ClusterUsersRecommender, MeanRecommender, PredictionType, RandomRecommender, UserBasedNeighborhoodRecommender
import clock
import errors

ROTATIONS = 8
error_function = errors.avg_miss


def main():
    clock.start("all testing")
    # user, item, rating
    X = ratings.read("train.csv")
    np.random.shuffle(X)

    mean_recommender = MeanRecommender()
    clock.start(f"testing {mean_recommender.name}")
    mean_error = errors.cross_validate(mean_recommender, X, ROTATIONS, error_function)
    clock.stop(f"testing {mean_recommender.name}")
    print(f"{mean_recommender.name} error: {mean_error}")

    random_recommender = RandomRecommender(max_rating=config.MAX_RATING)
    clock.start(f"testing {random_recommender.name}")
    random_error = errors.cross_validate(random_recommender, X, ROTATIONS, error_function)
    clock.stop(f"testing {random_recommender.name}")
    print(f"{random_recommender.name} error: {random_error}")

    # median number of common items is 4 -> discounted_similarity_threshold = 4
    # median similarity ~0.4
    # user_based_recommender = UserBasedNeighborhoodRecommender(k_neighbours=50, prediction_type=PredictionType.Z_SCORE, weight_items=True, min_similarity=0.4, beta=4)
    # clock.start(f"testing {user_based_recommender.name}")
    # user_based_error = errors.cross_validate(user_based_recommender, X, ROTATIONS, error_function)
    # clock.stop(f"testing {user_based_recommender.name}")
    # print(f"{user_based_recommender.name} error: {user_based_error}")

    cluster_users_recommender = ClusterUsersRecommender(k_neighbours=50, n_clusters=8, prediction_type=PredictionType.Z_SCORE, weight_items=True, min_similarity=0.4, beta=4)
    clock.start(f"testing {cluster_users_recommender.name}")
    user_based_error = errors.cross_validate(cluster_users_recommender, X, ROTATIONS, error_function)
    clock.stop(f"testing {cluster_users_recommender.name}")
    print(f"{cluster_users_recommender.name} error: {user_based_error}")

    clock.stop("all testing", print_time=True)


if __name__ == "__main__":
    main()
