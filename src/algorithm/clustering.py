from typing import Any, TypeAlias, Literal

import numpy as np
import numpy.typing as npt

AlgorithmT: TypeAlias = Literal["lloyds", "none"]


class ClusteringModel[DataT: Any]:
    """A clustering model for partitioning data into clusters using specified
    algorithms.

    Attributes:
        num_clusters (int): The number of clusters.
        cluster_centers (npt.NDArray[np.float64]): The coordinates of the
            cluster centers.
        method (AlgorithmT): The clustering method used.
        closest_centers (npt.NDArray[np.int64]): The indices of the closest
            cluster centers for each data point.
        error (np.float64): The clustering error.

    Examples:
        >>> import numpy as np
        >>> from clustering_model import ClusteringModel
        >>> data = np.random.rand(100, 2)  # 100 points in 2D space
        >>> model = ClusteringModel(data, num_clusters=3, method="lloyds")
        >>> model.train(data)
        >>> print(model.cluster_centers)
        >>> print(model.closest_centers)
        >>> print(model.error)
    """

    num_clusters: int
    cluster_centers: npt.NDArray[np.float64]
    method: AlgorithmT
    closest_centers: npt.NDArray[np.int64]
    error: np.float64

    def __init__(self, data: npt.NDArray[DataT], num_clusters: int, method: AlgorithmT) -> None:
        """Initializes the clustering model and assigns the random cluster
        centers.

        Args:
            data (npt.NDArray[DataT]): The input data for clustering. Must be a
                2-dimensional `numpy` array.
            num_clusters (int): The number of clusters.
            method (AlgorithmT): The clustering method to use.
        """
        self.num_clusters = num_clusters
        self.cluster_centers = self.get_random_cluster_centers(data, num_clusters)
        self.method = method

    @staticmethod
    def get_random_cluster_centers(data: npt.NDArray[DataT], num_clusters: int) -> npt.NDArray[DataT]:
        """Selects `num_cluster` amount random data points.

        Args:
            data (npt.NDArray[DataT]): The input data.
            num_clusters (int): The number of clusters.

        Returns:
            npt.NDArray[DataT]: Data points randomly chosen from the dataset.
        """
        return data[np.random.choice(data.shape[0], num_clusters, replace=False)]

    @staticmethod
    def get_distance(X: npt.NDArray[DataT], y: npt.NDArray[DataT]) -> npt.NDArray[np.float64]:
        """Calculates the Euclidean distance between a matrix and a vector or a
        vector and a vector.

        Args:
            X (npt.NDArray[DataT]): The input data points, matrix or vector.
            y (npt.NDArray[DataT]): The center point, vector.

        Returns:
            npt.NDArray[np.float64]: The distances from each vector in `X` to
            the vector `y`.
        """
        if len(X.shape) == 1:
            return np.array(np.linalg.norm(X - y))

        return np.array([np.linalg.norm(arr - y) for arr in X])

    def get_closest_centers(self, data: npt.NDArray[DataT]) -> npt.NDArray[np.int64]:
        """Finds the closest cluster center for each data point.

        Args:
            data (npt.NDArray[DataT]): The input data.

        Returns:
            npt.NDArray[np.int64]: The indices of the closest cluster centers.
        """
        return np.array([np.argmin(self.get_distance(self.cluster_centers, vector)) for vector in data])

    def get_centers_of_mass(
        self, data: npt.NDArray[DataT], closest_centers: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """Computes the center of mass for each cluster.

        Args:
            data (npt.NDArray[DataT]): The input data.
            closest_centers (npt.NDArray[np.int64]): The indices of the closest
                cluster centers.

        Returns:
            npt.NDArray[np.float64]: The new cluster centers.
        """
        return np.array([np.average(data[closest_centers == i, :], axis=0) for i in range(self.num_clusters)])

    def set_error(self, data: npt.NDArray[DataT]) -> np.float64:
        """Calculates and sets the clustering error (Within-Cluster Square of
        Sums).

        Args:
            data (npt.NDArray[DataT]): The input data.

        Returns:
            np.float64: The calculated error.
        """
        center_index: int
        wcss: np.float64 = np.float64(0)

        for center_index in range(self.num_clusters):
            subset: npt.NDArray[DataT] = data[self.closest_centers == center_index, :]
            for data_point in subset:
                wcss += np.sum((data_point - self.cluster_centers[center_index]) ** 2)

        self.error = wcss
        return wcss

    def train(self, data: npt.NDArray[DataT]) -> None:
        """Trains the clustering model using the specified method.

        Args:
            data (npt.NDArray[DataT]): The input data.
        """
        self.cluster_centers = self.get_random_cluster_centers(data, self.num_clusters)
        cluster_centers: npt.NDArray[np.float64]

        self.closest_centers = self.get_closest_centers(data)

        while not (
            (cluster_centers := self.get_centers_of_mass(data, self.closest_centers)) == self.cluster_centers
        ).all():
            self.cluster_centers = cluster_centers
            self.closest_centers = self.get_closest_centers(data)

        self.set_error(data)

    def train_step(self, data: npt.NDArray[DataT]) -> None:
        """Performs a single training step for the clustering model.

        Args:
            data (npt.NDArray[DataT]): The input data.
        """
        self.closest_centers = self.get_closest_centers(data)
        self.cluster_centers = self.get_centers_of_mass(data, self.closest_centers)
        self.closest_centers = self.get_closest_centers(data)
        self.set_error(data)
