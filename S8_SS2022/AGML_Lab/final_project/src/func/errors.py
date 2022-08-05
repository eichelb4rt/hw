import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from func.dynamic_line import Progress

from recommenders.recommender import Recommender
import func.clock as clock


def avg_miss(ratings, predicted_ratings):
    return np.mean(np.abs(ratings - predicted_ratings))


def rmse(ratings, predicted_ratings):
    return np.sqrt(np.mean(np.square(ratings - predicted_ratings)))


def cross_validate(recommender: Recommender, x: NDArray[np.int32], rotations=8, error_functions=[rmse, avg_miss], round_predictions=False) -> NDArray[np.float32]:
    n_samples = len(x)
    n_error_functions = len(error_functions)
    errors = np.empty((rotations, n_error_functions))
    progress = Progress(rotations, name=recommender.name)
    for i in range(rotations):
        test_mask = gen_test_mask(n_samples, rotations, i)
        test_samples = np.where(test_mask)[0]
        train_samples = np.where(1 - test_mask)[0]
        x_test = x[test_samples, :]
        x_train = x[train_samples, :]
        errors[i] = test_recommender(recommender, x_train, x_test, error_functions, round_predictions)
        progress.update(i + 1)
    clock.avg(f"{recommender.name} offline phase")
    clock.avg(f"{recommender.name} online phase")
    return np.mean(errors, axis=0)


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


def test_recommender(recommender: Recommender, x_train: NDArray[np.int32], x_test: NDArray[np.int32], error_functions=[rmse, avg_miss], round_predictions=False) -> NDArray[np.float32]:
    """Generates prediction errors for an array of error functions (for specific x_train, x_test)."""
    clock.start(f"{recommender.name} offline phase")
    recommender.fit(x_train)
    clock.stop(f"{recommender.name} offline phase")
    test_qualify, test_ratings = split_ratings(x_test)
    clock.start(f"{recommender.name} online phase")
    predictions = recommender.rate(test_qualify)
    clock.stop(f"{recommender.name} online phase")
    if round_predictions:
        predictions = np.round(predictions)
    return [error_function(test_ratings, predictions) for error_function in error_functions]


# [user, item, rating] -> [user, item], [rating]
def split_ratings(x_train: NDArray[np.int32]) -> Tuple[NDArray[np.int32], NDArray[np.int8]]:
    return x_train[:, :2], x_train[:, 2].astype(np.int8)


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
