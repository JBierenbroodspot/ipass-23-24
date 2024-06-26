import unittest
import sys

sys.path.append("./")

import numpy as np
import numpy.typing as npt
import numpy.testing as nptest

from src.algorithm.clustering import ClusteringModel

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
        np.random.seed(13)

        self.num_clusters = 2
        self.model = ClusteringModel(TEST_ARR, self.num_clusters, "none")
        # Seed 13 causes self.model.cluster_centers to be initially:
        # [[3, 5, 1, 4], [8, 1, 4, 6]]

    def test_get_random_cluster_centers(self) -> None:
        result: npt.NDArray[np.int64] = self.model.get_random_cluster_centers(TEST_ARR, self.num_clusters)

        self.assertNotIsInstance(result[1, 1], int)
        self.assertTrue(len(result) == self.num_clusters)
        with self.assertRaises(AssertionError):
            nptest.assert_array_equal(result[0], result[1])

    def test_init(self) -> None:
        self.assertEqual(self.model.num_clusters, self.num_clusters)
        self.assertEqual(len(self.model.cluster_centers), self.num_clusters)

        self.num_clusters = 3
        self.model: ClusteringModel[np.int64] = ClusteringModel(TEST_ARR, self.num_clusters, "none")

        self.assertEqual(self.model.num_clusters, self.num_clusters)
        self.assertEqual(len(self.model.cluster_centers), self.num_clusters)

    def test_get_distance(self) -> None:
        expected: npt.NDArray[np.float64] = np.array(
            [np.sqrt((1 - 3) ** 2 + (4 - 5) ** 2 + (6 - 1) ** 2 + (1 - 4) ** 2)]
        )

        result: npt.NDArray[np.float64] = self.model.get_distance(TEST_ARR[0], TEST_ARR[1])

        nptest.assert_array_equal(expected, result)

        expected = np.array(
            [
                np.sqrt((1 - 3) ** 2 + (4 - 5) ** 2 + (6 - 1) ** 2 + (1 - 4) ** 2),
                np.sqrt((3 - 3) ** 2 + (5 - 5) ** 2 + (1 - 1) ** 2 + (4 - 4) ** 2),
                np.sqrt((1 - 3) ** 2 + (5 - 5) ** 2 + (0 - 1) ** 2 + (1 - 4) ** 2),
                np.sqrt((8 - 3) ** 2 + (1 - 5) ** 2 + (4 - 1) ** 2 + (6 - 4) ** 2),
                np.sqrt((9 - 3) ** 2 + (9 - 5) ** 2 + (9 - 1) ** 2 + (9 - 4) ** 2),
            ]
        )

        result = self.model.get_distance(TEST_ARR, self.model.cluster_centers[0])

        nptest.assert_array_equal(expected, result)

    def test_get_closest_centers(self) -> None:
        print(self.model.cluster_centers)
        expected: npt.NDArray[np.int64] = np.array([0, 0, 0, 1, 1])

        result: npt.NDArray[np.int64] = self.model.get_closest_centers(TEST_ARR)

        nptest.assert_array_equal(expected, result)

    def test_get_centers_of_mass(self) -> None:
        expected: npt.NDArray[np.float64] = np.array(
            [
                [5 / 3, 14 / 3, 7 / 3, 6 / 3],
                [17 / 2, 10 / 2, 13 / 2, 15 / 2],
            ]
        )

        result: npt.NDArray[np.float64] = self.model.get_centers_of_mass(TEST_ARR, np.array([0, 0, 0, 1, 1]))

        nptest.assert_array_almost_equal_nulp(expected, result)
