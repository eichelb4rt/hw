from __future__ import annotations
import time
import numpy as np

import config
import func.ratings as ratings
from func.dynamic_line import Progress


def main():
    progress = Progress(8)
    for i in range(8):
        progress.update(i + 1)
        time.sleep(0.5)
    
    # X_TRAIN = ratings.read("train.csv")
    # users = ratings.user_map(X_TRAIN)
    # items = ratings.item_map(X_TRAIN)
    # n_users = len(users)
    # n_items = len(items)
    # ratings_matix = ratings.ratings_matrix(X_TRAIN, users, items)
    # rated_items = ratings_matix != config.MISSING_RATING

    # common_items = [np.count_nonzero(rated_items[u, :] & rated_items[v, :]) for u in range(n_users) for v in range(u, n_users)]
    # common_users = [np.count_nonzero(rated_items[i, :] & rated_items[j, :]) for i in range(n_items) for j in range(i, n_items)]
    # print(f"mean common items: {np.mean(common_items)}")
    # print(f"median common items: {np.median(common_items)}")
    # print(f"mean common users: {np.mean(common_users)}")
    # print(f"median common users: {np.median(common_users)}")


if __name__ == "__main__":
    main()
