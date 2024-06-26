import typing
from src.algorithm.clustering import ClusteringModel

import numpy as np
import numpy.typing as npt


class Elbow:
    min_k: int
    max_k: int
    data: npt.ArrayLike
    model: ClusteringModel[typing.Any]
    k_scores: npt.NDArray[np.float64]
    distances: npt.NDArray[np.float64]

    def __init__(
        self,
        model: ClusteringModel[typing.Any],
        min_k: int,
        max_k: int,
        data: npt.ArrayLike,
    ) -> None:
        self.min_k = min_k
        self.max_k = max_k
        self.data = data
        self.model = model
        self.k_scores = np.zeros((max_k - min_k,), np.float64)
        self.distances = np.zeros((max_k - min_k), np.float64)

    def get_k_scores(self) -> npt.NDArray[np.float64]:
        i: int
        for i, k in enumerate(range(self.min_k, self.max_k)):
            model = type(self.model)(self.data, k, self.model.method)  # type: ignore
            model.train(self.data)  # type: ignore
            self.k_scores[i] = model.error

        return self.k_scores

    def get_intercept_line(self) -> npt.NDArray[np.float64]:
        delta_x: int = self.max_k - self.min_k
        delta_y = self.k_scores[-1] - self.k_scores[0]
        gradient = delta_y / delta_x
        y_intercept = gradient * self.min_k - self.k_scores[0]

        return np.array(
            [gradient * i - y_intercept for i in range(self.min_k, self.max_k)]
        )

    def get_intercept_distances(self) -> npt.NDArray[np.float64]:
        intercept_line: npt.NDArray[np.float64] = self.get_intercept_line()

        distances: npt.NDArray[np.float64] = np.zeros(len(intercept_line))

        for i in range(len(intercept_line)):
            distances[i] = np.abs(intercept_line[i] - self.k_scores[i])

        return distances

    def find_elbow(self) -> np.int64:
        self.get_k_scores()
        self.distances = self.get_intercept_distances()

        return np.argmax(self.distances)
