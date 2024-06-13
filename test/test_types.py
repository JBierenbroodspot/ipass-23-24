import unittest

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt

from src.types import OneDArray, TwoDArray
from src.exceptions import DimensionError


@unittest.skip("Deprecated")
class TestOneDArray(unittest.TestCase):
    @staticmethod
    def test_successful_use() -> None:
        one_d_array: npt.NDArray = np.array([1, 2, 3, 4])

        result: OneDArray = OneDArray(one_d_array)

        nptest.assert_array_equal(one_d_array, result())

    def test_unsuccessful_use(self) -> None:
        two_d_array: npt.NDArray = np.array([[1, 2], [1, 2]])

        with self.assertRaises(DimensionError):
            OneDArray(two_d_array)


@unittest.skip("Deprecated")
class TestTwoDArray(unittest.TestCase):
    def test_one_d_array(self) -> None:
        one_d_array: npt.NDArray = np.array([1, 2, 3, 4])

        with self.assertRaises(DimensionError):
            TwoDArray(one_d_array)

    @staticmethod
    def test_two_d_array() -> None:
        two_d_array: npt.NDArray = np.array([[1, 2], [1, 2]])

        result: TwoDArray = TwoDArray(two_d_array)

        nptest.assert_array_equal(two_d_array, result())

    def test_three_d_array(self) -> None:
        three_d_array: npt.NDArray = np.zeros(shape=(3, 3, 3))

        with self.assertRaises(DimensionError):
            TwoDArray(three_d_array)
