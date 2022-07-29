import numpy as np
from recommender import *


def append_recommendations(X_qualify, recommendations):
    return np.append(X_qualify, recommendations, axis=1)


def main():
    # user, item
    X_qualify = np.genfromtxt("qualifying_blanc.csv",
                              delimiter=",", dtype=np.int32)
    # user, item, rating
    X_train = np.genfromtxt("train.csv", delimiter=",", dtype=np.int32)

    mean_recommender = MeanRecommender().fit(X_train)
    random_recommender = RandomRecommender(max_rating=5)

    np.savetxt("qualifying_mean.csv", append_recommendations(X_qualify, mean_recommender.recommend(X_qualify)),
               delimiter=",", newline="\n", encoding="utf-8")
    np.savetxt("qualifying_random.csv", append_recommendations(X_qualify, random_recommender.recommend(X_qualify)),
               delimiter=",", newline="\n", encoding="utf-8")


if __name__ == "__main__":
    main()
