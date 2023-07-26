import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from jax import Array
from jax import numpy as jnp
from jax import random as jrnd


class SyntheticDataset(ABC):
    """Abstract class for synthetic datasets."""

    @abstractmethod
    def generate(self) -> Tuple[Array, Array]:
        pass


class SyntheticClusters(SyntheticDataset):
    """Class for generating synthetic clusters."""

    def __init__(
        self,
        n_samples_per_cluster: int,
        n_features: int,
        cluster_centers: Array,
        cluster_std: Array,
        random_seed: Optional[int] = None,
    ):
        """Creates a synthetic dataset for clustering.

        Args:
            n_samples_per_cluster (int): The number of samples in each cluster.
            n_features (int): Number of features each sample has.
            cluster_centers (Array): Array of shape (n_clusters, n_features) containing the cluster centers.
            cluster_std (Array): Standard deviations of the clusters.
            random_seed (int, optional): Random seed used in any operations involving randomness, such that the results are reproducible. Defaults to None.
        """
        self.n_samples_per_cluster = n_samples_per_cluster
        self.n_features = n_features
        self.cluster_centers = cluster_centers
        self.cluster_std = cluster_std
        self.random_seed = random_seed or int(time.time())

    def generate(self) -> Tuple[Array, Array]:
        """Creates a synthetic dataset for clustering.

        Each sample is assigned to a cluster based on the cluster centers and the cluster standard deviation.
        Cluster data points are sampled from a normal distribution with mean equal to the cluster center and
        standard deviation equal to the cluster standard deviation.

        Returns:
            Tuple[Array, Array]: Tuple containing the data and the cluster labels.
                The data is an array of shape (n_samples, n_features) and the cluster labels
                is an array of shape (n_samples, ).
        """

        key = jrnd.PRNGKey(self.random_seed)
        n_clusters = self.cluster_centers.shape[0]
        n_samples = self.n_samples_per_cluster * n_clusters
        data = jnp.zeros((n_samples, self.n_features))
        labels = jnp.zeros((n_samples,))

        for cluster in range(n_clusters):
            start_idx = cluster * self.n_samples_per_cluster
            end_idx = (cluster + 1) * self.n_samples_per_cluster

            # cluster_samples = jrnd.multivariate_normal(
            #     key,
            #     self.cluster_centers[cluster, :],
            #     jnp.eye(self.n_features) * (self.cluster_std[cluster] ** 2),
            #     (self.n_samples_per_cluster, )
            # )

            # N(mu, sigma^2) = sigma * N(0, 1) + mu
            cluster_samples = (
                jrnd.normal(
                    key,
                    (self.n_samples_per_cluster, self.n_features),
                )
                * self.cluster_std[cluster]
                + self.cluster_centers[cluster, :]
            )

            data = data.at[start_idx:end_idx].set(cluster_samples)
            labels = labels.at[start_idx:end_idx].set(cluster)

        return data, labels
