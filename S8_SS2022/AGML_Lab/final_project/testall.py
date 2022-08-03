import numpy as np
import config
import ratings
from recommender import ClusterItemsRecommender, ClusterUsersRecommender, ItemBasedNeighborhoodRecommender, MeanRecommender, PredictionType, RandomRecommender, Recommender, UserBasedNeighborhoodRecommender
import clock
import errors

ROTATIONS = 8
error_function = errors.avg_miss

def test(recommender: Recommender, X):
    clock.start(f"testing {recommender.name}")
    mean_error = errors.cross_validate(recommender, X, ROTATIONS, error_function)
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
    item_based_recommender = ItemBasedNeighborhoodRecommender(k_neighbours=50, weight_items=True, min_similarity=0.5, beta=6)
    test(item_based_recommender, X)
    
    # NOTE: Clustering items lead to incredibly worse results (0.25 -> 0.3) and didn't make it much faster (44s -> 30s)
    # cluster_items_recommender = ClusterItemsRecommender(k_neighbours=50, n_clusters=4, weight_items=True, min_similarity=0.4, beta=4)
    # test(cluster_items_recommender, X)

    clock.stop("all testing", print_time=True)


if __name__ == "__main__":
    main()
