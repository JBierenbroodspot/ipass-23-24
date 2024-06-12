import unittest
import operator

import numpy as np
import numpy.testing as nptest

from src.algorithm.clustering import ClusteringModel
from src.types import TwoDArray, OneDArray

np.random.seed(1)

TEST_TWO_D_ARR: TwoDArray[int] = TwoDArray(
    np.array(
        [
            [1, 4, 6, 1],
            [3, 5, 1, 4],
            [1, 5, 0, 1],
            [8, 1, 4, 6],
            [9, 9, 9, 9],
        ]
    )
)


class TestClusteringModel(unittest.TestCase):
    def test_get_random_cluster_centers(self) -> None:
        num_clusters: int = 2
        result: np.ndarray = ClusteringModel.get_random_cluster_centers(
            TEST_TWO_D_ARR, num_clusters
        )()

        self.assertNotIsInstance(result[1, 1], int)
        self.assertTrue(len(result) == num_clusters)
        with self.assertRaises(AssertionError):
            nptest.assert_array_equal(result[0], result[1])

    def test_init(self) -> None:
        num_clusters: int = 2
        model: ClusteringModel = ClusteringModel(TEST_TWO_D_ARR, num_clusters)

        self.assertEqual(model.num_clusters, num_clusters)
        self.assertNotEqual(model.cluster_centers, None)

    def test_get_distance(self) -> None:
        expected: float = np.sqrt(
            (1 - 3) ** 2 + (4 - 5) ** 2 + (6 - 1) ** 2 + (1 - 4) ** 2
        )

        result: float = ClusteringModel.get_distance(
            OneDArray(TEST_TWO_D_ARR()[0]), OneDArray(TEST_TWO_D_ARR()[1])
        )

        self.assertEqual(expected, result)
