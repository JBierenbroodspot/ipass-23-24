import typing

import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
import numpy.typing as npt


class Subplot[AxT: matplotlib.axes.Axes | list[matplotlib.axes.Axes] | list[list[matplotlib.axes.Axes]]](
    typing.NamedTuple
):
    fig: matplotlib.figure.Figure
    ax: AxT
    n_cols: int
    n_rows: int


def create_subplot() -> Subplot[matplotlib.axes.Axes]:
    return Subplot(*plt.subplots(), n_cols=1, n_rows=1)  # type: ignore


def create_subplot_row(n_cols: int) -> Subplot[list[matplotlib.axes.Axes]]:
    fig: matplotlib.figure.Figure
    ax: list[matplotlib.axes.Axes]

    fig, ax = plt.subplots(ncols=n_cols)  # type: ignore

    return Subplot(fig, ax, n_cols=n_cols, n_rows=1)


def create_subplot_matrix(n_cols: int, n_rows: int) -> Subplot[list[list[matplotlib.axes.Axes]]]:
    fig: matplotlib.figure.Figure
    ax: list[list[matplotlib.axes.Axes]]

    fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows)  # type: ignore

    return Subplot(fig, ax, n_cols=n_cols, n_rows=n_rows)


def plot_graph(
    ax: matplotlib.axes.Axes,
    x_axis: npt.ArrayLike,
    y_axis: npt.ArrayLike,
    **kwargs: typing.Any,
) -> None:
    ax.plot(x_axis, y_axis, **kwargs)  # type: ignore


def plot_scatter(
    ax: matplotlib.axes.Axes,
    x_axis: npt.ArrayLike,
    y_axis: npt.ArrayLike,
    **kwargs: typing.Any,
) -> None:
    ax.scatter(x_axis, y_axis, **kwargs)  # type: ignore


if __name__ == "__main__":
    matplotlib.interactive(True)
    curve_y: npt.NDArray[np.float64] = np.array([32, 16, 8, 4, 2, 1, 0.5, 0.25])
    x_axis: npt.NDArray[np.int64] = np.arange(len(curve_y))

    subplots: Subplot[matplotlib.axes.Axes] = create_subplot()
    plot_graph(subplots.ax, x_axis=x_axis, y_axis=curve_y)
    plt.show()  # type: ignore
