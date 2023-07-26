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
        """Update the cluster centroids based on the data.
        Returns the within-cluster sum of squares for each cluster.

        Args:
            data (Array): Array of shape (n_samples, n_features)
        """
        for cluster in range(self.n_clusters):
            predicted_labels = self.predict(data)
            cluster_samples = data[predicted_labels == cluster, :]
            self.cluster_centroids = self.cluster_centroids.at[cluster].set(
                cluster_samples.mean(axis=0)
            )

    def update(self, data: Array, n_iterations: int = 1) -> Array:
        """Update the cluster centroids based on the data.

        Args:
            data (Array): Array of shape (n_samples, n_features)
            n_iterations (int, optional): number of times to perform centroid update. Defaults to 1.

        Returns:
            Array: an array of shape (n_iterations, n_clusters) containing the within-cluster sum of squares
                for each cluster for each iteration.
        """
        wcss = jax.numpy.zeros((n_iterations, 1))
        for iteration in range(n_iterations):
            self._update(data)
            wcss_iteration = self.compute_wcss(data)
            wcss = wcss.at[iteration, :].set(wcss_iteration)

        return wcss

    def compute_wcss(self, data: Array) -> Array:
        """Computes the within-cluster sum of squares.

        Args:
            data (Array): Array of shape (n_samples, n_features)

        Returns:
            Array: an array of shape (1, ) containing the within-cluster sum of squares.
        """

        # TODO: optimize

        wcss: Array = jax.numpy.zeros((1,))
        predicted_labels = self.predict(data)
        for cluster in range(self.n_clusters):
            cluster_samples = data[predicted_labels == cluster, :]

            if cluster_samples.shape[0] == 0:
                continue

            cluster_mean = cluster_samples.mean(axis=0)

            # vector representing the distance of each sample to the cluster mean
            dist_vector = cluster_samples - cluster_mean

            # square of magnitude of the distance vector for each sample
            abs_dist_square = jax.numpy.sum(dist_vector * dist_vector, axis=1)

            # actual cluster wcss is the sum of the squares of the distances
            wcss += jax.numpy.sum(abs_dist_square)

        return wcss

    def compute_silhouette_score(self, data: Array) -> Array:
        """Computes the 3 silhouette scores for the clustering.

        Args:
            data (Array): Array of shape (n_samples, n_features).

        Returns:
            Array:
        """

        #
        predicted_labels = self.predict(data)
        mean_intra_cluster_distances = jax.numpy.zeros((data.shape[0],))

        for cluster in range(self.n_clusters):
            cluster_sample_idx = jax.numpy.where(predicted_labels == cluster)[0]
            cluster_samples = data[cluster_sample_idx, :]

            for sample_idx in cluster_sample_idx:
                # data[sample_idx, :] is the sample from cluster ``cluster`` for
                # which we are computing the mean distance to every other sanmple in the cluster
                sample_distances = jax.numpy.linalg.norm(
                    cluster_samples - data[sample_idx, :], axis=1
                )
                # we subtract 1 because the sample at sample_idx is not included in the mean
                mean_distance_to_sample = jax.numpy.sum(sample_distances) / (
                    cluster_samples.shape[0] - 1
                )
                mean_intra_cluster_distances = mean_intra_cluster_distances.at[
                    sample_idx
                ].set(mean_distance_to_sample)

        return mean_intra_cluster_distances
