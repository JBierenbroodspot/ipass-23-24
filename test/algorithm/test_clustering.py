import unittest

import numpy as np
import numpy.typing as npt
import numpy.testing as nptest

from src.algorithm.clustering import ClusteringModel

np.random.seed(1)

TEST_ARR: npt.NDArray[np.int64] = np.array(
    [
        [1, 4, 6, 1],
        [3, 5, 1, 4],
        [1, 5, 0, 1],
        [8, 1, 4, 6],
        [9, 9, 9, 9],
    ]
)


class TestClusteringModel(unittest.TestCase):
    model: ClusteringModel[np.int64]
    num_clusters: int

    def setUp(self) -> None:
        self.num_clusters = 2
        self.model = ClusteringModel(TEST_ARR, self.num_clusters)

    def test_get_random_cluster_centers(self) -> None:
        result: npt.NDArray[np.int64] = self.model.get_random_cluster_centers(
            TEST_ARR, self.num_clusters
        )

        self.assertNotIsInstance(result[1, 1], int)
        self.assertTrue(len(result) == self.num_clusters)
        with self.assertRaises(AssertionError):
            nptest.assert_array_equal(result[0], result[1])

    def test_init(self) -> None:
        self.assertEqual(self.model.num_clusters, self.num_clusters)
        self.assertEqual(len(self.model.cluster_centers), self.num_clusters)

        self.num_clusters = 3
        self.model: ClusteringModel[np.int64] = ClusteringModel(
            TEST_ARR, self.num_clusters
        )

        self.assertEqual(self.model.num_clusters, self.num_clusters)
        self.assertEqual(len(self.model.cluster_centers), self.num_clusters)

    def test_get_distance_vector_to_vector(self) -> None:
        expected: float = np.sqrt(
            (1 - 3) ** 2 + (4 - 5) ** 2 + (6 - 1) ** 2 + (1 - 4) ** 2
        )

        result: npt.NDArray[np.float64] = ClusteringModel[np.int64].get_distance(
            TEST_ARR[0], TEST_ARR[1]
        )

        self.assertEqual(expected, result)
