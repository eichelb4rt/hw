import config
import ratings
from recommender import MeanRecommender, RandomRecommender


def main():
    # user, item
    X_QUALIFY = ratings.read("qualifying_blanc.csv")
    # user, item, rating
    X_TRAIN = ratings.read("train.csv")
    
    ratings.plot_rating_frequency(X_TRAIN)

    # mean_recommender = MeanRecommender().fit(X_TRAIN)
    # mean_ratings = ratings.generate_ratings(mean_recommender, X_QUALIFY)
    # ratings.save("qualifying_mean.csv", mean_ratings)

    # random_recommender = RandomRecommender(max_rating=config.MAX_RATING)
    # random_ratings = ratings.generate_ratings(random_recommender, X_QUALIFY)
    # ratings.save("qualifying_random.csv", random_ratings)


if __name__ == "__main__":
    main()
