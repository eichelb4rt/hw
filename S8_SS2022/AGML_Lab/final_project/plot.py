import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import config
import ratings
from recommender import UserBasedNeighborhoodRecommender
import errors


def frequency(rating_counts: Counter, x_name="item", save_file=None):
    plt.clf()
    # sort them into an array
    sorted_counts = np.array([count for _, count in rating_counts.most_common()])

    # plot them
    n_items = len(sorted_counts)
    plt.plot(range(n_items), sorted_counts)
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
    plt.xlabel(f"ratings")
    plt.ylabel("number of ratings")
    if save_file is not None:
        plt.savefig(os.path.join(config.PLOT_DIR, save_file))
    else:
        plt.show()


def main():
    x_train = ratings.read("train.csv")
    rating_frequency(x_train, "train_frequency.png")
    predictions = np.round(ratings.read_output("qualifying_cluster_users.csv"))
    rating_frequency(predictions, "prediction_frequency.png")
    # x_qualify = ratings.read("qualifying_blanc.csv")
    # recommender = UserBasedNeighborhoodRecommender(k=50, min_similarity=-np.infty).fit(x_train)
    # predictions = np.round(ratings.generate_ratings(recommender, x_qualify))
    # rating_frequency(predictions, "prediction_frequency.png")


if __name__ == "__main__":
    main()
