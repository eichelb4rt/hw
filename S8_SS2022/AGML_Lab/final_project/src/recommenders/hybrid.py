from __future__ import annotations
import os
import numpy as np
from enum import Enum
from nptyping import NDArray
import matplotlib.pyplot as plt

import config
import func.errors as errors
import func.ratings as ratings
from recommenders.recommender import Recommender


class MinimizationGoal(Enum):
    MEAN_ABSOLUTE_ERROR = 1
    MEAN_SQUARED_ERROR = 2


class Hybrid(Recommender):
    name = "hybrid"

    def __init__(self, recommenders: list[Recommender], min_goal=MinimizationGoal.MEAN_ABSOLUTE_ERROR, held_out_partitions=4, epsilon=1e-3, lr=0.5, max_iterations=100, plot_descent=False):
        self.recommenders = recommenders
        self.n_recommenders = len(recommenders)
        # R, where size of whole training set / size of held out partition = R
        self.held_out_partitions = held_out_partitions
        self.min_goal = min_goal
        self.epsilon = epsilon
        self.lr = lr
        self.max_iterations = max_iterations
        # recommender weights
        self.alpha: NDArray[np.float32] = None
        self.plot_descent = plot_descent

    def fit(self, x_train) -> Hybrid:
        # fit alpha using gradient descent
        self.alpha = self.fit_alpha(x_train)
        # fit every recommender on the training
        for recommender in self.recommenders:
            recommender.fit(x_train)
        return self

    def fit_alpha(self, x_train):
        x = np.copy(x_train)
        np.random.shuffle(x)

        fit_ratings, held_out_ratings = ratings.split_train_test(x, self.held_out_partitions, 1)
        for recommender in self.recommenders:
            recommender.fit(fit_ratings)
        x_held_out, y_held_out = ratings.split_ratings(held_out_ratings)

        # determine alpha with gradient descent on held_out
        # only have to do that once
        single_predictions = self.single_ratings(x_held_out)
        # init alpha = 1 / n
        alpha = np.full(self.n_recommenders, 1 / self.n_recommenders)
        iteration = 0
        error = np.infty
        if self.plot_descent:
            error_history = []

        while iteration < self.max_iterations:
            # predict
            combined_predictions = alpha @ single_predictions
            # check error here, so we don't have to predict twice
            last_error = error
            error = self.error(y_held_out, combined_predictions)
            if self.plot_descent:
                error_history.append(error)
            # if we're still improving by a lot, go on
            if np.abs(last_error - error) < self.epsilon:
                break
            # descent
            alpha -= self.lr * self.gradient(y_held_out, combined_predictions, single_predictions)
            iteration += 1

        if self.plot_descent:
            plot_error_history(error_history, save_file="descent.png")
        return alpha

    def rate(self, x_qualify) -> Recommender:
        return np.clip(self.alpha @ self.single_ratings(x_qualify), config.MIN_RATING, config.MAX_RATING)

    def single_ratings(self, x_qualify):
        return np.array([recommender.rate(x_qualify) for recommender in self.recommenders])

    def error(self, y_held_out, combined_prediction) -> float:
        if self.min_goal == MinimizationGoal.MEAN_ABSOLUTE_ERROR:
            return errors.mae(y_held_out, combined_prediction)
        if self.min_goal == MinimizationGoal.MEAN_SQUARED_ERROR:
            return errors.mse(y_held_out, combined_prediction)

    def gradient(self, y_held_out, combined_prediction, single_predictions):
        if self.min_goal == MinimizationGoal.MEAN_ABSOLUTE_ERROR:
            return (single_predictions @ np.sign(combined_prediction - y_held_out)) / len(y_held_out)
        if self.min_goal == MinimizationGoal.MEAN_SQUARED_ERROR:
            return (single_predictions @ (combined_prediction - y_held_out)) / len(y_held_out)


def plot_error_history(error_hist, save_file=None):
    plt.clf()
    n_iterations = len(error_hist)
    plt.plot(range(n_iterations), error_hist)
    # plt.ylim(ymin=0)
    plt.xlabel("iteration")
    plt.ylabel("error")
    if save_file is not None:
        plt.savefig(os.path.join(config.PLOT_DIR, save_file))
    else:
        plt.show()