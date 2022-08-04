from __future__ import annotations
from numpy.typing import NDArray
import numpy as np

import func.clock as clock
import config
import func.distance as distance


class KMeans:
    def __init__(self, k, distance_measure, epsilon=0.05, max_iterations=20):
        self.k = k
        # prototype vectors, centroids of their cells
        self.prototypes: NDArray[np.float32] = None
        self.distance_measure = distance_measure
        # change of prototypes < epsilon: end iterations
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        # number of dimensions
        self.n_dims = 0

    def fit(self, vectors: NDArray[np.int8]) -> KMeans:
        self.n_dims = vectors.shape[1]
        self.init_prototypes()
        changes = np.full(self.k, np.infty)
        iteration = 0
        while np.sum(changes) >= self.epsilon and iteration < self.max_iterations:
            labels = self.label_all(vectors)
            old_prototypes = self.prototypes
            self.prototypes = self.get_prototypes(vectors, labels)
            changes = [self.distance_measure(old_prototypes[c], self.prototypes[c]) for c in range(self.k)]
            iteration += 1
        return self

    def init_prototypes(self):
        self.prototypes = np.random.randint(config.MIN_RATING, config.MAX_RATING + 1, size=(self.k, self.n_dims), dtype=np.int8)

    def label_all(self, vectors: NDArray[np.int8]) -> NDArray[np.int8]:
        return np.array([self.label(vector) for vector in vectors], dtype=np.int8)

    def get_prototypes(self, vectors: NDArray[np.int8], labels: NDArray[np.int8]) -> NDArray[np.float32]:
        prototypes = np.empty((self.k, self.n_dims))
        for c in range(self.k):
            class_vectors = vectors[labels == c]
            prototypes[c] = self.mean(class_vectors)
        return prototypes

    def mean(self, vectors: NDArray[np.int8]) -> NDArray[np.float32]:
        if len(vectors) == 0:
            return np.full(self.n_dims, config.MISSING_RATING)
        mean_vector = np.empty(self.n_dims)
        # mask where the ratings are not missing
        rated = vectors != config.MISSING_RATING
        # dimensions where all ratings are missing -> missing rating
        rated_dimensions = np.count_nonzero(rated, axis=0) > 0
        mean_vector[~rated_dimensions] = config.MISSING_RATING
        mean_vector[rated_dimensions] = np.mean(vectors[:, rated_dimensions], axis=0, where=rated[:, rated_dimensions])
        return mean_vector

    def label(self, vector: NDArray[np.int8]) -> np.int8:
        distances = [self.distance_measure(vector, prototype) for prototype in self.prototypes]
        return np.argmin(distances)


def main():
    n_clusters = 20
    n_items = 1000
    n_users = 1000
    vectors = np.random.randint(-1, 5, size=(n_users, n_items))

    clock.start("clustering")
    kmeans = KMeans(k=n_clusters, distance_measure=distance.manhattan, epsilon=0.05, max_iterations=10).fit(vectors)
    clock.stop("clustering", print_time=True)

    print(kmeans.prototypes)
    labels = kmeans.label_all(vectors)
    for c in range(kmeans.k):
        print(f"class {c}: {len(vectors[labels == c])}")


if __name__ == "__main__":
    main()
