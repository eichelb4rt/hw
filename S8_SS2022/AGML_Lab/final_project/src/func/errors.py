import numpy as np
from numpy.typing import NDArray
from func.dynamic_line import Progress

import func.ratings as ratings
import func.clock as clock
from recommenders.recommender import Recommender


def mae(ratings, predicted_ratings):
    """mean absolute error"""
    return np.mean(np.abs(ratings - predicted_ratings))


def rmse(ratings, predicted_ratings):
    """root mean squared error"""
    return np.sqrt(np.mean(np.square(ratings - predicted_ratings)))


def mse(ratings, predicted_ratings):
    """mean squared error"""
    return np.mean(np.square(ratings - predicted_ratings))


# NOTE: i wrote my own cross validation so that i could use different error functions
def cross_validate(recommender: Recommender, x: NDArray[np.int32], rotations=8, error_functions=[rmse, mae], round_predictions=False) -> NDArray[np.float32]:
    n_error_functions = len(error_functions)
    errors = np.empty((rotations, n_error_functions))
    progress = Progress(rotations, name=recommender.name)
    for i in range(rotations):
        x_train, x_test = ratings.split_train_test(x, rotations, i)
        errors[i] = test_recommender(recommender, x_train, x_test, error_functions, round_predictions)
        progress.update(i + 1)
    clock.avg(f"{recommender.name} offline phase")
    clock.avg(f"{recommender.name} online phase")
    return np.mean(errors, axis=0)


def test_recommender(recommender: Recommender, x_train: NDArray[np.int32], x_test: NDArray[np.int32], error_functions=[rmse, mae], round_predictions=False) -> NDArray[np.float32]:
    """Generates prediction errors for an array of error functions (for specific x_train, x_test)."""
    clock.start(f"{recommender.name} offline phase")
    recommender.fit(x_train)
    clock.stop(f"{recommender.name} offline phase")
    test_qualify, test_ratings = ratings.split_ratings(x_test)
    clock.start(f"{recommender.name} online phase")
    predictions = recommender.rate(test_qualify)
    clock.stop(f"{recommender.name} online phase")
    if round_predictions:
        predictions = np.round(predictions)
    return [error_function(test_ratings, predictions) for error_function in error_functions]


def loocv(estimator, samples, labels, error_function):
    # TODO: Implement leave one out crossvalidation for Ridge Regression
    m = len(samples)
    errors = [None] * m
    # https://stackoverflow.com/a/28057966/12795023
    train_indices = np.arange(1, m) - np.tri(m, m - 1, k=-1, dtype=bool)
    test_indices = [[i] for i in np.arange(m)]
    # build training/testing samples/labels
    train_samples = samples[train_indices]
    train_labels = labels[train_indices]
    test_samples = samples[test_indices]
    test_labels = labels[test_indices]
    # train, predict, calculate mean RMSE
    for i in range(m):
        estimator.fit(train_samples[i], train_labels[i])
        prediction = estimator.predict(test_samples[i])
        errors[i] = error_function(test_labels[i], prediction)
    return np.mean(errors)
