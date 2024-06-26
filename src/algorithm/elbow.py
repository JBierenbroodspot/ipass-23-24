import typing
from src.algorithm.clustering import ClusteringModel

import numpy as np
import numpy.typing as npt


class Elbow:
    """The Elbow class is used to determine the optimal number of clusters (k)
    for a clustering model using the elbow method.

    Attributes:
        min_k (int): The minimum number of clusters to evaluate.
        max_k (int): The maximum number of clusters to evaluate.
        data (npt.ArrayLike): The input data for clustering.
        model (ClusteringModel[typing.Any]): The clustering model.
        k_scores (npt.NDArray[np.float64]): The scores for each k value.
        distances (npt.NDArray[np.float64]): The distances from the k scores to
            the intercept line.

    Examples:
        >>> import numpy as np
        >>> from src.algorithm.clustering import ClusteringModel
        >>> from src.algorithm.elbow import Elbow
        >>> data = np.random.rand(100, 2)  # 100 points in 2D space
        >>> model = ClusteringModel(data, num_clusters=3, method="lloyds")  # num clusters does not matter here
        >>> elbow = Elbow(model, min_k=1, max_k=10, data=data)
        >>> optimal_k = elbow.find_elbow()
        >>> print(optimal_k)
    """

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
        """Initializes the Elbow class.

        Args:
            model (ClusteringModel[typing.Any]): The clustering model.
            min_k (int): The minimum number of clusters to evaluate.
            max_k (int): The maximum number of clusters to evaluate.
            data (npt.ArrayLike): The input data for clustering.
        """
        self.min_k = min_k
        self.max_k = max_k
        self.data = data
        self.model = model
        self.k_scores = np.zeros((max_k - min_k,), np.float64)
        self.distances = np.zeros((max_k - min_k), np.float64)

    def get_k_scores(self) -> npt.NDArray[np.float64]:
        """Calculates the k scores for each number of clusters from `min_k` to
        `max_k`.

        Returns:
            npt.NDArray[np.float64]: The k scores for each k value.
        """
        i: int
        for i, k in enumerate(range(self.min_k, self.max_k)):
            model = type(self.model)(self.data, k, self.model.method)  # type: ignore
            model.train(self.data)  # type: ignore
            self.k_scores[i] = model.error

        return self.k_scores

    def get_intercept_line(self) -> npt.NDArray[np.float64]:
        """Calculates the intercept line between the first and last k scores.

        Returns:
            npt.NDArray[np.float64]: The intercept line values.
        """
        delta_x: int = self.max_k - self.min_k
        delta_y = self.k_scores[-1] - self.k_scores[0]
        gradient = delta_y / delta_x
        y_intercept = gradient * self.min_k - self.k_scores[0]

        return np.array([gradient * i - y_intercept for i in range(self.min_k, self.max_k)])

    def get_intercept_distances(self) -> npt.NDArray[np.float64]:
        """
        Calculates the distances from the k scores to the intercept line.

        Returns:
            npt.NDArray[np.float64]: The distances from the k scores to the
                intercept line.
        """
        intercept_line: npt.NDArray[np.float64] = self.get_intercept_line()

        distances: npt.NDArray[np.float64] = np.zeros(len(intercept_line))

        for i in range(len(intercept_line)):
            distances[i] = np.abs(intercept_line[i] - self.k_scores[i])

        return distances

    def find_elbow(self) -> np.int64:
        """Finds the optimal number of clusters (k) using the elbow method.

        Returns:
            np.int64: The optimal number of clusters.
        """
        self.get_k_scores()
        self.distances = self.get_intercept_distances()

        return np.argmax(self.distances)
