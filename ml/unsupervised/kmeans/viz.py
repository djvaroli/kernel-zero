from typing import Optional

import jax
import matplotlib.pyplot as plt
from jax import Array


def plot_clusters2d(
    data: Array,
    labels: Array,
    centroid_coords: Optional[Array] = None,
):
    """Given a 2D array of data samples and associated lable,s plots the data samples
    and colors them according to their cluster label. If centroid coordinates are provided,
    plots the centroid coordinates as well.

    Args:
        data (Array): 2D array of shape (n_samples, 2)
        labels (Array): 1D array of shape (n_samples, ) containing cluster labels
        centroid_coords (Optional[Array], optional): 2D array of shape (n_clusters, 2) containing
            centroid coordinates. Defaults to None.

    Raises:
        ValueError: if data is not 2D.
    """
    if data.shape[1] != 2:
        raise ValueError("Can only plot 2D data")

    if centroid_coords is not None:
        if centroid_coords.shape[1] != 2:
            raise ValueError("Can only plot 2D data")

        n_clusters = len(centroid_coords)
    else:
        # assumes all clusters have at least one sample
        n_clusters = len(jax.numpy.unique(labels))

    for cluster in range(n_clusters):
        cluster_samples = data[labels == cluster, :]
        plt.scatter(cluster_samples[:, 0], cluster_samples[:, 1])

        cluster_mean = cluster_samples.mean(axis=0)
        plt.scatter(
            cluster_mean[0],
            cluster_mean[1],
            marker="*",
            label=f"Mean of cluster {cluster}",
        )

        if centroid_coords is not None:
            cluster_centroid = centroid_coords[cluster]
            plt.scatter(
                cluster_centroid[0],
                cluster_centroid[1],
                marker="x",
                label=f"Centroid of cluster {cluster}",
            )

    plt.legend()
