import unittest

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt

from src.types import OneDArray
from src.exceptions import DimensionError


class TestOneDArray(unittest.TestCase):
    def test_successful_use(self) -> None:
        one_d_array: npt.NDArray = np.array([1, 2, 3, 4])

        result: OneDArray = OneDArray(one_d_array)

        nptest.assert_array_equal(one_d_array, result())

    def test_unsuccessful_use(self) -> None:
        two_d_array: npt.NDArray = np.array([[1, 2], [1, 2]])

        with self.assertRaises(DimensionError):
            OneDArray(two_d_array)
