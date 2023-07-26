from abc import ABC, abstractmethod

from attrs import define
from jax import Array
from jax import numpy as jnp


class Metric(ABC):
    """Abstract class for metrics"""

    @property
    def name(self) -> str:
        return self.__class__.__name__


class SilhouetteScore(Metric):
    """Computes the silhouette score for a clustering."""

    def __init__(
        self,
        return_mean: bool = True,
    ) -> None:
        """Creates a silhouette score metric.

        Args:
            return_mean (bool, optional): Whether to return the mean silhouette score or the silhouette score for each sample. Defaults to True.
        """
        self.return_mean = return_mean

    def __call__(self, data: Array, labels: Array) -> Array:
        """Computes the silhouette score for a clustering.

        Args:
            data (Array): Array of shape (n_samples, n_features).
            labels (Array): Array of shape (n_samples, ) containing the predicted labels.

        Returns:
            Array: Array of shape (1, ) containing the averaged silhouette score,
                or an array of shape (n_samples, ) containing the silhouette score for each sample.
        """
        element_wise_silhouette_scores = self._compute(data, labels)
        if self.return_mean:
            return jnp.mean(element_wise_silhouette_scores)

        return element_wise_silhouette_scores

    def _compute(self, data: Array, labels: Array) -> Array:
        """Computes the mean intra-cluster distance for each sample.

        Args:
            data (Array): Array of shape (n_samples, n_features).
            labels (Array): Array of shape (n_samples, ) containing the predicted labels.

        Returns:
            Array: Array of shape (n_samples, ) containing the mean intra-cluster distance for each sample.

        Raises:
            ValueError: If the number of clusters is 0.
        """
        scores = jnp.zeros((data.shape[0],))
        n_clusters = jnp.unique(labels).shape[0]

        if n_clusters == 0:
            raise ValueError("Number of clusters cannot be 0")

        if n_clusters == 1:
            return

        for sample_idx in range(data.shape[0]):
            # these are the a(i) and b(i) terms in the silhouette score formula
            mean_within_cluster_dist = jnp.inf
            smallest_mean_between_cluster_dist = jnp.inf

            # compute the distance of the sample to every other sample in the cluster
            distances_to_sample = jnp.linalg.norm(data - data[sample_idx, :], axis=1)

            # loop over each cluster and compute the a(i) and b(i) terms
            for cluster in range(n_clusters):
                in_cluster_sample_idxs = jnp.where(labels == cluster)[0]
                sum_to_cluster_dist = jnp.sum(
                    distances_to_sample[in_cluster_sample_idxs]
                )

                # if we are looking at same cluster as that of the sample then compute a(i)
                # otherwise compute b(i)
                if cluster == labels[sample_idx]:
                    mean_within_cluster_dist = sum_to_cluster_dist / (
                        in_cluster_sample_idxs.shape[0] - 1
                    )

                else:
                    mean_between_cluster_dist = (
                        sum_to_cluster_dist / in_cluster_sample_idxs.shape[0]
                    )
                    smallest_mean_between_cluster_dist = jnp.minimum(
                        smallest_mean_between_cluster_dist, mean_between_cluster_dist
                    )

            # renaming for convenience, but longer name is more descriptive
            a_i = mean_within_cluster_dist
            b_i = smallest_mean_between_cluster_dist

            # compute the silhouette score for the sample
            scores = scores.at[sample_idx].set((b_i - a_i) / jnp.maximum(b_i, a_i))

        return scores
