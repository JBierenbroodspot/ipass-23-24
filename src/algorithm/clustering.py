from typing import TypeVar, Union

import numpy as np
import numpy.typing as npt

from src.types import OneDArray, TwoDArray

DataT: TypeVar = TypeVar("DataT", *np.ScalarType)


class ClusteringModel:
    num_clusters: int
    cluster_centers: TwoDArray[Union[DataT, np.float64]]

    def __init__(self, data: TwoDArray[DataT], num_clusters: int) -> None:
        self.num_clusters = num_clusters
        self.cluster_centers = self.get_random_cluster_centers(data, num_clusters)

    @staticmethod
    def get_random_cluster_centers(
        data: TwoDArray[DataT], num_clusters: int
    ) -> TwoDArray:
        return TwoDArray(
            data()[np.random.choice(data().shape[0], num_clusters, replace=False)]
        )
