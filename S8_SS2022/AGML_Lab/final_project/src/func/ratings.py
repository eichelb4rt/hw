import os
import numpy as np
from typing import Tuple
from numpy.typing import NDArray

import config
import func.clock as clock
from recommenders.recommender import Recommender


def read(filename: str) -> NDArray[np.int32]:
    return np.genfromtxt(os.path.join(config.INPUT_DIR, filename), delimiter=",", dtype=np.int32)


def read_output(filename: str) -> NDArray[np.int32]:
    return np.genfromtxt(os.path.join(config.OUTPUT_DIR, filename), delimiter=",", dtype=np.int32)


def fit_and_save(recommender: Recommender, x_train: NDArray[np.int32], x_qualify: NDArray[np.int32], round_predictions=False):
    clock.start(f"{recommender.name} offline phase")
    recommender.fit(x_train)
    clock.stop(f"{recommender.name} offline phase", print_time=True)
    clock.start(f"{recommender.name} online phase")
    predictions = generate_ratings(recommender, x_qualify)
    clock.stop(f"{recommender.name} online phase", print_time=True)
    if round_predictions:
        predictions = np.round(predictions)
    save(f"qualifying_{recommender.name}.csv", predictions, int_predictions=round_predictions)


def save(filename: str, ratings: NDArray[np.float32], int_predictions=False):
    fmt = ["%d", "%d", "%d"] if int_predictions else ["%d", "%d", "%.2f"]
    np.savetxt(os.path.join(config.OUTPUT_DIR, filename), ratings, fmt=fmt, delimiter=",", newline="\n", encoding="utf-8")


def generate_ratings(recommender: Recommender, x_qualify: NDArray[np.int32]) -> NDArray[np.float32]:
    return append_ratings(x_qualify, recommender.rate(x_qualify))


def append_ratings(x_qualify: NDArray[np.int32], recommendations: NDArray[np.int32]):
    recommendations_column = np.array([recommendations]).T
    return np.append(x_qualify, recommendations_column, axis=1)


# [user, item, rating] -> [user, item], [rating]
def split_ratings(x_train: NDArray[np.int32]) -> Tuple[NDArray[np.int32], NDArray[np.int8]]:
    return x_train[:, :2], x_train[:, 2].astype(np.int8)


def split_train_test(x, rotations, iteration):
    n_samples = len(x)
    test_mask = gen_test_mask(n_samples, rotations, iteration)
    test_samples = np.where(test_mask)[0]
    train_samples = np.where(~test_mask)[0]
    x_test = x[test_samples, :]
    x_train = x[train_samples, :]
    return x_train, x_test


def gen_test_mask(n_samples, rotations, iteration):
    samples_per_rotation = n_samples // rotations
    start_idx = samples_per_rotation * iteration
    end_idx = samples_per_rotation * (iteration + 1)
    mask = np.full(n_samples, False)

    last_iteration = iteration == rotations - 1
    if not last_iteration:
        mask[start_idx: end_idx] = True
    else:
        mask[start_idx:] = True
    return mask


def user_map(x_train: NDArray[np.int32]) -> dict[int, int]:
    """Makes a map user ID -> index of user.
    """

    users = {}
    index = 0
    for user, _, _ in x_train:
        if user not in users:
            users[user] = index
            index += 1
    return users


def item_map(x_train: NDArray[np.int32]) -> dict[int, int]:
    """Makes a map item ID -> index of item.
    """

    items = {}
    index = 0
    for _, item, _ in x_train:
        if item not in items:
            items[item] = index
            index += 1
    return items


def ratings_matrix(x_train: NDArray[np.int32], user_map: dict[int, int], item_map: dict[int, int]) -> NDArray[np.int8]:
    n_users = len(user_map)
    n_items = len(item_map)
    # array of missing ratings
    ratings = np.full((n_users, n_items), config.MISSING_RATING, dtype=np.int8)
    # fill ratings where found
    for user, item, rating in x_train:
        idx_user = user_map[user]
        idx_item = item_map[item]
        ratings[idx_user, idx_item] = rating
    return ratings
