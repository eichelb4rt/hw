import os
import numpy as np
from collections import Counter
from numpy.typing import NDArray
import matplotlib.pyplot as plt

import config
from recommender import Recommender


def read(filename: str) -> NDArray[np.int32]:
    return np.genfromtxt(os.path.join(config.INPUT_DIR, filename), delimiter=",", dtype=np.int32)


def save(filename: str, ratings: NDArray[np.int32]):
    np.savetxt(os.path.join(config.OUTPUT_DIR, filename), ratings, delimiter=",", newline="\n", encoding="utf-8")


def generate_ratings(recommender: Recommender, x_qualify: NDArray[np.int32]) -> NDArray[np.int32]:
    return append_ratings(x_qualify, recommender.rate(x_qualify))


def append_ratings(x_qualify: NDArray[np.int32], recommendations: NDArray[np.int32]):
    return np.append(x_qualify, recommendations, axis=1)


def plot_frequency(rating_counts: Counter, x_name="item"):
    # sort them into an array
    sorted_counts = np.array([count for _, count in rating_counts.most_common()])

    # plot them
    n_items = len(sorted_counts)
    plt.plot(range(n_items), sorted_counts)
    plt.ylim(ymin=0)
    plt.xlabel(f"{x_name} index")
    plt.ylabel("number of ratings")
    plt.show()


def plot_item_frequency(x_train):
    # count ratings
    rating_counts = Counter()
    for _, item, _ in x_train:
        rating_counts[item] += 1

    # plot frequency
    plot_frequency(rating_counts, x_name="item")


def plot_user_frequency(x_train):
    # count ratings
    rating_counts = Counter()
    for user, _, _ in x_train:
        rating_counts[user] += 1

    # plot frequency
    plot_frequency(rating_counts, x_name="user")


def plot_rating_frequency(x_train):
    # count ratings
    rating_counts = Counter()
    for _, _, rating in x_train:
        rating_counts[rating] += 1
    
    possible_ratings = range(config.MIN_RATING, config.MAX_RATING + 1)
    sorted_counts = [rating_counts[rating] for rating in possible_ratings]

    # plot them
    plt.bar(possible_ratings, sorted_counts)
    plt.ylim(ymin=0)
    plt.xlabel(f"ratings")
    plt.ylabel("number of ratings")
    plt.show()
