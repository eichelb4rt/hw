import numpy as np
from recommender import MeanRecommender, RandomRecommender
import config


def append_recommendations(x_qualify, recommendations):
    return np.append(x_qualify, recommendations, axis=1)


def main():
    # user, item
    X_QUALIFY = np.genfromtxt("qualifying_blanc.csv",
                              delimiter=",", dtype=np.int32)
    # user, item, rating
    X_TRAIN = np.genfromtxt("train.csv", delimiter=",", dtype=np.int32)

    mean_recommender = MeanRecommender().fit(X_TRAIN)
    random_recommender = RandomRecommender(max_rating=config.MAX_RATING)

    np.savetxt("qualifying_mean.csv", append_recommendations(X_QUALIFY, mean_recommender.recommend(X_QUALIFY)), delimiter=",", newline="\n", encoding="utf-8")
    np.savetxt("qualifying_random.csv", append_recommendations(X_QUALIFY, random_recommender.recommend(X_QUALIFY)), delimiter=",", newline="\n", encoding="utf-8")


if __name__ == "__main__":
    main()
