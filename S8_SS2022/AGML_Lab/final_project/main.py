import config
import ratings
from recommender import MeanRecommender, RandomRecommender
import clock


def main():
    # user, item
    X_QUALIFY = ratings.read("qualifying_blanc.csv")
    # user, item, rating
    X_TRAIN = ratings.read("train.csv")

    clock.start("mean_recommender")
    mean_recommender = MeanRecommender().fit(X_TRAIN)
    mean_ratings = ratings.generate_ratings(mean_recommender, X_QUALIFY)
    ratings.save("qualifying_mean.csv", mean_ratings)
    clock.stop("mean_recommender")

    clock.start("random_recommender")
    random_recommender = RandomRecommender(max_rating=config.MAX_RATING)
    random_ratings = ratings.generate_ratings(random_recommender, X_QUALIFY)
    ratings.save("qualifying_random.csv", random_ratings)
    clock.stop("random_recommender")

    clock.start("user_based_recommender")
    user_based_recommender = RandomRecommender(max_rating=config.MAX_RATING)
    user_based_ratings = ratings.generate_ratings(user_based_recommender, X_QUALIFY)
    ratings.save("qualifying_user_based.csv", user_based_ratings)
    clock.stop("user_based_recommender")


if __name__ == "__main__":
    main()
