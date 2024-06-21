from typing import Any, TypeAlias, Literal

import numpy as np
import numpy.typing as npt

AlgorithmT: TypeAlias = Literal["lloyds", "none"]


class ClusteringModel[DataT: Any]:
    num_clusters: int
    cluster_centers: npt.NDArray[np.float64]
    method: AlgorithmT
    closest_centers: npt.NDArray[np.int64]
    error: np.float64

    def __init__(
        self, data: npt.NDArray[DataT], num_clusters: int, method: AlgorithmT
    ) -> None:
        self.num_clusters = num_clusters
        self.cluster_centers = self.get_random_cluster_centers(data, num_clusters)
        self.method = method

    @staticmethod
    def get_random_cluster_centers(
        data: npt.NDArray[DataT], num_clusters: int
    ) -> npt.NDArray[DataT]:
        return data[np.random.choice(data.shape[0], num_clusters, replace=False)]

    @staticmethod
    def get_distance(
        X: npt.NDArray[DataT], y: npt.NDArray[DataT]
    ) -> npt.NDArray[np.float64]:
        if len(X.shape) == 1:
            return np.array(np.linalg.norm(X - y))

        return np.array([np.linalg.norm(arr - y) for arr in X])

    def get_closest_centers(self, data: npt.NDArray[DataT]) -> npt.NDArray[np.int64]:
        return np.array(
            [
                np.argmin(self.get_distance(self.cluster_centers, vector))
                for vector in data
            ]
        )

    def get_centers_of_mass(
        self, data: npt.NDArray[DataT], closest_centers: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        return np.array(
            [
                np.average(data[closest_centers == i, :], axis=0)
                for i in range(self.num_clusters)
            ]
        )

    def set_error(self, data: npt.NDArray[DataT]) -> np.float64:
        center_index: int
        wcss: np.float64 = np.float64(0)

        for center_index in range(self.num_clusters):
            subset: npt.NDArray[DataT] = data[self.closest_centers == center_index, :]
            for data_point in subset:
                wcss += np.sum((data_point - self.cluster_centers[center_index]) ** 2)

        self.error = wcss
        return wcss

    def train(self, data: npt.NDArray[DataT]) -> None:
        self.cluster_centers = self.get_random_cluster_centers(data, self.num_clusters)
        cluster_centers: npt.NDArray[np.float64]

        self.closest_centers = self.get_closest_centers(data)

        while not (
            (cluster_centers := self.get_centers_of_mass(data, self.closest_centers))
            == self.cluster_centers
        ).all():
            self.cluster_centers = cluster_centers
            self.closest_centers = self.get_closest_centers(data)

        self.set_error(data)

    def train_step(self, data: npt.NDArray[DataT]) -> None:
        self.closest_centers = self.get_closest_centers(data)
        self.cluster_centers = self.get_centers_of_mass(data, self.closest_centers)
        self.closest_centers = self.get_closest_centers(data)
