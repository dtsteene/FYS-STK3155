from pathlib import Path
import numpy as np
from matplotlib import colormaps, pyplot as plt, cm
import matplotlib as mpl


# Set up for LaTeX rendering
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["figure.titlesize"] = 15


def setColors(
    variable_arr: np.ndarray,
    cmap_name: str = "viridis",
    norm_type: str = "log",
) -> tuple[mpl.colors.Colormap, mpl.colors.LogNorm, mpl.cm.ScalarMappable]:
    """
    Returns a colormap, a normalization instance, and a scalar mappable instance.

    Args:
        variable_arr (np.ndaray):
            Array of values to be plotted.
        cmap_name (str, optional):
            Name of the colormap. Defaults to "viridis".
        norm_type (str, optional):
            Type of normalization to use. Defaults to "log".

    Returns:
        tuple[mpl.colors.Colormap, mpl.colors.LogNorm, mpl.cm.ScalarMappable]:
            A tuple containing the colormap, normalization instance, and scalar mappable instance.
    """
    cmap = colormaps.get_cmap(cmap_name)
    if norm_type == "log":
        norm = mpl.colors.LogNorm(vmin=np.min(variable_arr), vmax=np.max(variable_arr))
    elif norm_type == "linear":
        norm = mpl.colors.Normalize(
            vmin=np.min(variable_arr), vmax=np.max(variable_arr)
        )
    else:
        raise ValueError(f"Invalid norm_type: {norm_type}")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    return cmap, norm, sm


def setup_axis(xlim: tuple[int], ylim: tuple[int]) -> plt.Axes:
    """Set up the axis for a function plot.

    Args:
        xlim (tuple[int]): The limits of the x-axis.
        ylim (tuple[int]): The limits of the y-axis.

    Returns:
        plt.Axes: The axis for the plot.
    """
    _, ax = plt.subplots()

    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    for s in ax.spines.values():
        s.set_zorder(0)

    return ax


def set_plot_limits(ax, xlim=None, ylim=None, zlim=None):
    """
    Set the limits for the x, y, and z axes of a Matplotlib plot.

    Parameters:
    - ax: The axis object of the plot.
    - xlim: Tuple containing the lower and upper limits for the x-axis, e.g., (xmin, xmax).
    - ylim: Tuple containing the lower and upper limits for the y-axis, e.g., (ymin, ymax).
    - zlim: Tuple containing the lower and upper limits for the z-axis, e.g., (zmin, zmax).
    """

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)


def plot_at_timestep(
    x: np.ndarray,
    res_dnn: np.ndarray,
    res_analytic: np.ndarray,
    t: float,
    func_name: str,
    save: bool,
    savePath: Path = None,
    saveName: str = None,
) -> None:
    plt.figure(figsize=plt.figaspect(0.5))
    plt.title(f"{func_name}: Computed solutions at time = {t:g}")
    plt.plot(x, res_dnn, label="Deep neural network")
    plt.plot(x, res_analytic, label="Analytical")
    plt.xlabel("Position $x$")
    plt.legend()
    if save:
        plt.savefig(savePath / f"{saveName}_timestep_{t}.pdf", bbox_inches="tight")
    plt.close()


def plot_at_timestepEuler(
    x_numeric: np.ndarray,
    res_numeric: np.ndarray,
    x_anal: np.ndarray,
    res_analytic: np.ndarray,
    t: float,
    func_name: str,
    save: bool,
    savePath: Path = None,
    saveName: str = None,
) -> None:
    plt.figure(figsize=plt.figaspect(0.5))
    plt.title(f"{func_name}: Computed solutions at time = {t:g}")
    plt.plot(x_anal, res_analytic, label="Analytical")
    plt.plot(x_numeric, res_numeric, label="Forward Euler")
    plt.xlabel("Position $x$")
    plt.legend()
    if save:
        plt.savefig(savePath / f"{saveName}_timestep_{t}.pdf", bbox_inches="tight")
    plt.close()


def plot_surface(
    T: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    title: str,
    save: bool,
    savePath: Path = None,
    saveName: str = None,
) -> None:
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(projection="3d")
    ax.set_title(title)
    ax.plot_surface(T, X, Z, linewidth=0, antialiased=False, cmap=cm.viridis)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Position $x$")
    if save:
        plt.savefig(savePath / f"{saveName}.pdf", bbox_inches="tight")
    plt.close()
