import numpy as np
import numpy.typing as npt

from .exceptions import DimensionError


class OneDArray[DTypeT: npt.DTypeLike]:
    _arr: npt.NDArray[DTypeT]

    def __init__(self, arr: npt.NDArray[DTypeT]) -> None:
        if len(arr.shape) > 1:
            raise DimensionError(
                "Trying to assign multidimensional array to OneDArray."
            )

        self._arr = arr

    def __call__(self) -> npt.NDArray[DTypeT]:
        return self._arr
