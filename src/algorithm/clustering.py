from typing import Any

import numpy as np
import numpy.typing as npt

# from src.types import OneDArray, TwoDArray

# DataT: TypeVar = TypeVar("DataT", *np.ScalarType)


class ClusteringModel[DataT: Any]:
    num_clusters: int
    cluster_centers: npt.NDArray[np.float64]

    def __init__(self, data: npt.NDArray[DataT], num_clusters: int) -> None:
        self.num_clusters = num_clusters
        self.cluster_centers = self.get_random_cluster_centers(data, num_clusters)

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
