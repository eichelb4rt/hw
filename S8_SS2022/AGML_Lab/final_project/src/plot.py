import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import Counter

import config
import func.ratings as ratings
from func.dynamic_line import Progress

from recommenders.simple import RandomRecommender, MeanRecommender
from recommenders.cluster_items import ClusterItemsRecommender
from recommenders.cluster_users import ClusterUsersRecommender
from recommenders.factorization import ALS
from recommenders.hybrid import Hybrid, MinimizationGoal
from recommenders.item_based import ItemBasedNeighborhoodRecommender
from recommenders.recommender import Recommender
from recommenders.user_based import PredictionType, UserBasedNeighborhoodRecommender


def frequency(rating_counts: Counter, x_name="item", save_file=None):
    plt.clf()
    # sort them into an array
    sorted_counts = np.array([count for _, count in rating_counts.most_common()])

    # plot them
    n_items = len(sorted_counts)
    plt.plot(range(n_items), sorted_counts)
    plt.title(f"Number of ratings per {x_name}")
    plt.ylim(ymin=0)
    plt.xlabel(f"{x_name} index")
    plt.ylabel("number of ratings")
    if save_file is not None:
        plt.savefig(os.path.join(config.PLOT_DIR, save_file))
    else:
        plt.show()


def item_frequency(samples, save_file=None):
    # count ratings
    rating_counts = Counter()
    for _, item, _ in samples:
        rating_counts[item] += 1

    # plot frequency
    frequency(rating_counts, x_name="item", save_file=save_file)


def user_frequency(samples, save_file=None):
    # count ratings
    rating_counts = Counter()
    for user, _, _ in samples:
        rating_counts[user] += 1

    # plot frequency
    frequency(rating_counts, x_name="user", save_file=save_file)


def rating_frequency(samples, save_file=None):
    plt.clf()
    # count ratings
    rating_counts = Counter()
    for _, _, rating in samples:
        rating_counts[rating] += 1

    possible_ratings = range(config.MIN_RATING, config.MAX_RATING + 1)
    sorted_counts = [rating_counts[rating] for rating in possible_ratings]

    # plot them
    plt.bar(possible_ratings, sorted_counts)
    plt.ylim(ymin=0)
    plt.xlabel("ratings")
    plt.ylabel("number of ratings")
    if save_file is not None:
        plt.savefig(os.path.join(config.PLOT_DIR, save_file))
    else:
        plt.show()


def plot_filled(recommenders: list[Recommender], x_train):
    user_map = ratings.user_map(x_train)
    item_map = ratings.item_map(x_train)
    combinations = np.array([[user, item] for user in user_map.keys() for item in item_map.keys()])
    ratings_matrix = ratings.ratings_matrix(x_train, user_map, item_map)

    titles = ["original"]
    matrices = [ratings_matrix]

    for recommender in recommenders:
        full_matrix = gen_filled(recommender, x_train, combinations, user_map, item_map, ratings_matrix)
        matrices.append(full_matrix)
        titles.append(recommender.name)

    # cmap = ListedColormap([(1, 0, 0), (1, 0.6, 0), (0, 1, 0), (0, 0.6, 1), (0, 0, 1)])
    cmap = cm.get_cmap("cividis").copy()
    cmap.set_under("white")

    for title, matrix in zip(titles, matrices):
        plt.clf()
        plt.title(title)
        plt.axis("off")
        im = plt.imshow(matrix, cmap=cmap)
        im.set_clim(config.MIN_RATING, config.MAX_RATING)
        plt.colorbar(im, fraction=0.04)
        plt.savefig(os.path.join(config.PLOT_DIR, f"matrix_{title}.png"), bbox_inches='tight')


def gen_filled(recommender: Recommender, x_train, combinations, user_map, item_map, ratings_matrix):
    recommender.fit(x_train)
    print(recommender.name)

    predictions = recommender.rate(combinations)

    full_ratings = ratings_matrix.copy()
    for prediction, (user, item) in zip(predictions, combinations):
        u = user_map[user]
        i = item_map[item]
        full_ratings[u, i] = prediction
    return full_ratings


def main():
    x_train = ratings.read("train.csv")

    # item_based_recommender = ItemBasedNeighborhoodRecommender(k_neighbours=50, weight_items=True, min_similarity=0.5, beta=6)
    # cluster_users_recommender = ClusterUsersRecommender(k_neighbours=50, n_clusters=8, prediction_type=PredictionType.Z_SCORE, weight_items=True, min_similarity=0.4, beta=4)
    # als_recommender = ALS(latent_dimensions=20, regularization_factor=5, epsilon=1e-1, max_iterations=10)
    # hybrid_recommender = Hybrid([cluster_users_recommender, item_based_recommender, als_recommender],
    #                             min_goal=MinimizationGoal.MEAN_SQUARED_ERROR, epsilon=1e-4, lr=1, max_iterations=100, plot_descent=True)
    # plot_filled([item_based_recommender, cluster_users_recommender, als_recommender, hybrid_recommender], x_train)

    user_frequency(x_train, "user_frequency.png")
    item_frequency(x_train, "item_frequency.png")

    # rating_frequency(x_train, "train_frequency.png")
    # predictions = np.round(ratings.read_output("qualifying_hybrid.csv"))
    # rating_frequency(predictions, "prediction_frequency.png")
    # x_qualify = ratings.read("qualifying_blanc.csv")
    # predictions = np.round(ratings.generate_ratings(hybrid_recommender.fit(x_train), x_qualify))
    # rating_frequency(predictions, "prediction_frequency.png")


if __name__ == "__main__":
    main()
