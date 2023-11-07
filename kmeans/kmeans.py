"""Define the KMeans class."""
import numpy as np


class Kmeans:
    """K-means clustering algorithm."""

    def __init__(self, n_clusters=4, max_iter=300, tolerance=1e-4):
        """
        Initialize the K-means clustering algorithm.

        Parameters
        ----------
            n_clusters : int
                Number of clusters.
            max_iter : int
                Maximum number of iterations.
            tolerance : float
                Tolerance for the stopping criteria.
            centroids : np.ndarray
                Centroids of the clusters.
            data_clusters : list
                List of data points in each cluster.
            data_errors : list
                List of errors of data points in each cluster.
            centroids_errors : list
                List of errors of centroids in each cluster.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.centroids = None
        self.data_clusters = [[] for _ in range(self.n_clusters)]
        self.data_errors = [[] for _ in range(self.n_clusters)]
        self.centroids_errors = [[] for _ in range(self.n_clusters)]

    def create_clusters(self, centroids, X):
        """Create clusters."""
        data_clusters = [[] for _ in range(self.n_clusters)]
        for i, x in enumerate(X):
            distances = [np.linalg.norm(x - c) for c in centroids]
            cluster = np.argmin(distances)
            data_clusters[cluster].append(i)
        return data_clusters

    def get_centroids(self, data_clusters, X):
        """return centroids of clusters."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i, cluster in enumerate(data_clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    def _get_centroids_error(self, centroids, centroids_old):
        """
        Return the error between the old and new centroids.
        """
        return np.linalg.norm(centroids - centroids_old)

    def _get_data_error(self, centroid, data_clusters):
        """
        Return the error between the data points and its centroid.
        """
        return np.linalg.norm(data_clusters - centroid)

    def fit(self, X):
        """Fit the K-means clustering algorithm."""
        self.centroids = X[
            np.random.choice(X.shape[0], self.n_clusters, replace=False)
        ]
        for _ in range(self.max_iter):
            self.data_clusters = self._create_clusters(self.centroids, X)
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.data_clusters, X)
            for i in range(self.n_clusters):
                self.centroids_errors[i].append(
                    self._get_centroids_error(
                        self.centroids[i], centroids_old[i]
                    )
                )
                self.data_errors[i].append(
                    self._get_data_error(
                        self.centroids[i], X[self.data_clusters[i]]
                    )
                )