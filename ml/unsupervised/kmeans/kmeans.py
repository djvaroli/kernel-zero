import time
from typing import Optional

import jax
from jax import Array


class KMeans:
    def __init__(self, n_clusters: int, random_seed: Optional[int] = None) -> None:
        """KMeans clustering algorithm.

        Args:
            n_clusters (int): number of clusters to form.
            random_seed (Optional[int], optional): Random seed used in any operations involving
                randomness, such that the results are reproducible. Defaults to None.
        """
        self.n_clusters = n_clusters
        self.cluster_centroids: Array
        self._random_seed = random_seed or int(time.time())

    def initialize(self, data: Array) -> None:
        """Initializes the cluster centroids according to the data.

        Args:
            data (Array): Array of shape (n_samples, n_features)
        """
        self._initialize_centroids(data.min(), data.max(), data.shape[1])

    def _initialize_centroids(
        self, minval: float, maxval: float, n_features: int
    ) -> None:
        key = jax.random.PRNGKey(self._random_seed)
        self.cluster_centroids = jax.random.uniform(
            key, (self.n_clusters, n_features), minval=minval, maxval=maxval
        )

    def predict(self, data: Array) -> Array:
        """Predictes the cluster labels for each sample in the data.

        Args:
            data (Array): Array of shape (n_samples, n_features)

        Returns:
            Array: Array of shape (n_samples, ) containing the cluster labels for each sample.
        """
        # [i, j] corresponds to distance of sample i to cluster j
        distance_matrix = jax.numpy.zeros((data.shape[0], self.n_clusters))

        for cluster in range(self.n_clusters):
            l2_norm = jax.numpy.linalg.norm
            distance_matrix = distance_matrix.at[:, cluster].set(
                l2_norm(data - self.cluster_centroids[cluster], axis=1)
            )

        return distance_matrix.argmin(axis=1)

    def _update(self, data: Array) -> None:
        for cluster in range(self.n_clusters):
            predicted_labels = self.predict(data)
            cluster_samples = data[predicted_labels == cluster, :]
            self.cluster_centroids = self.cluster_centroids.at[cluster].set(
                cluster_samples.mean(axis=0)
            )

    def update(self, data: Array, n_iterations: int = 1) -> None:
        """Update the cluster centroids based on the data.

        Args:
            data (Array): Array of shape (n_samples, n_features)
            n_iterations (int, optional): number of times to perform centroid update. Defaults to 1.
        """
        for _ in range(n_iterations):
            self._update(data)
