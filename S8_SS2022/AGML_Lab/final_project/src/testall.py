import numpy as np

import config
import func.ratings as ratings
import func.clock as clock
import func.errors as errors

from recommenders.recommender import Recommender
from recommenders.simple import MeanRecommender, RandomRecommender
from recommenders.user_based import UserBasedNeighborhoodRecommender
from recommenders.item_based import ItemBasedNeighborhoodRecommender
from recommenders.cluster_users import ClusterUsersRecommender
from recommenders.cluster_items import ClusterItemsRecommender
from recommenders.factorization import ALS

ROTATIONS = 8
error_function = errors.avg_miss


def test(recommender: Recommender, x):
    clock.start(f"testing {recommender.name}")
    mean_error = errors.cross_validate(recommender, x, ROTATIONS, error_function)
    clock.stop(f"testing {recommender.name}")
    print(f"{recommender.name} error: {mean_error}\n")


def main():
    clock.start("all testing")
    # user, item, rating
    X = ratings.read("train.csv")
    np.random.shuffle(X)

    mean_recommender = MeanRecommender()
    test(mean_recommender, X)

    random_recommender = RandomRecommender(max_rating=config.MAX_RATING)
    test(random_recommender, X)

    # median number of common items is 4 -> discounted_similarity_threshold = 4
    # median similarity ~0.4
    # user_based_recommender = UserBasedNeighborhoodRecommender(k_neighbours=50, prediction_type=PredictionType.Z_SCORE, weight_items=True, min_similarity=0.4, beta=4)
    # test(user_based_recommender, X)

    # cluster_users_recommender = ClusterUsersRecommender(k_neighbours=50, n_clusters=8, prediction_type=PredictionType.Z_SCORE, weight_items=True, min_similarity=0.4, beta=4)
    # test(cluster_users_recommender, X)

    # (min_simlarity, error): (0.3, 0.276), (0.4, 0.259), (0.5, 0.250), (0.6, 0.262), (0.7, 0.308)
    # (beta, error): (4, 0.250), (6, 0.250), (8, 0.249), (10, 0.2489), (12, 0.250), (14, 0.249), (16, 0.249)
    # item_based_recommender = ItemBasedNeighborhoodRecommender(k_neighbours=50, weight_items=True, min_similarity=0.5, beta=6)
    # test(item_based_recommender, X)

    # NOTE: Clustering items lead to incredibly worse results (0.25 -> 0.3) and didn't make it much faster (44s -> 30s)
    # cluster_items_recommender = ClusterItemsRecommender(k_neighbours=50, n_clusters=4, weight_items=True, min_similarity=0.4, beta=4)
    # test(cluster_items_recommender, X)

    # (max_iteratins, error): (40, 0.281), (5, 0.288)
    # (regularization_factor, error): (1e-2, 0.288), (1e-1, 0.288), (1, 0.269), (3, 0.266), (5, 0.265), (10, 0.268)
    # (latent_dimensions, error): (5, 0.302), (10, 0.281), (20, 0.265), (30, 0.258)
    # latent_dimensions=10, regularization_factor=2, epsilon=1e-1, max_iterations=10 -> super fast, error 2.8
    # latent_dimensions=15, regularization_factor=3, epsilon=1e-1, max_iterations=10 -> medium speed, error 2.67
    # latent_dimensions=20, regularization_factor=5, epsilon=1e-1, max_iterations=10 -> meh speed, error 2.63
    als_recommender = ALS(latent_dimensions=8, regularization_factor=5, epsilon=1e-2, max_iterations=20)
    test(als_recommender, X)

    clock.stop("all testing", print_time=True)


if __name__ == "__main__":
    main()
