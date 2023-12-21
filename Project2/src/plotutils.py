import matplotlib as mpl
from matplotlib import colormaps, pyplot as plt
import numpy as np
from pathlib import Path
from typing import Callable, Optional
import seaborn as sns
from pandas import DataFrame
from NeuralNetwork import NeuralNet
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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
        variable_arr (np.ndarray):
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


def PlotPredictionPerVariable(
    x_vals: np.ndarray,
    y_vals: list[np.ndarray],
    variable_arr: np.ndarray,
    target_func: Optional[Callable] = None,
    x_label: str = r"$x$",
    y_label: str = r"$y$",
    variable_label: str = r"$\eta$",
    variable_type: str = "log",
    title: str = "Predicted polynomials",
    colormap: str = "viridis",
    n_epochs: int = 500,
    savePlots: bool = False,
    showPlots: bool = True,
    figsPath: Path = None,
    saveName: str = None,
) -> None:
    r"""
    Plots predicted polynomials for different values of a variable parameter.

    Args:
        x_vals (np.ndarray):
            Array of x values to plot.
        y_vals (List[np.ndarray]):
            List of arrays of y values to plot, one for each value of the variable parameter.
        variable_arr (np.ndarray):
            Array of values of the variable parameter.
        target_func (Callable, optional):
            Function representing the true target function to plot. Defaults to None.
        x_label (str, optional):
            Label for the x-axis. Defaults to r"$x$".
        y_label (str, optional):
            Label for the y-axis. Defaults to r"$y$".
        variable_label (str, optional):
            Label for the variable parameter. Defaults to r"$\eta$".
        variable_type (str, optional):
            Type of normalization to use. Defaults to "log".
        title (str, optional):
            Title for the plot. Defaults to "Predicted polynomials".
        n_epochs (int, optional):
            Number of epochs used for training the model. Defaults to 500.
        savePlots (bool, optional):
            Whether to save the plot as a PNG file. Defaults to False.
        showPlots (bool, optional):
            Whether to display the plot. Defaults to True.
        figsPath (Path, optional):
            Path to the directory where the plot should be saved. Defaults to None.
        saveName (str, optional):
            Name of the file to save the plot as. Defaults to None.

    Returns:
        None
    """
    fig, ax = plt.subplots()

    cmap, norm, sm = setColors(
        variable_arr, cmap_name=colormap, norm_type=variable_type
    )

    ax.set_xlim(np.nanmin(x_vals), np.nanmax(x_vals))
    ax.set_ylim(np.nanmin(y_vals), np.nanmax(y_vals))

    for i, ynew in enumerate(y_vals):
        ax.plot(x_vals, ynew, color=cmap(norm(variable_arr[i])))

    if target_func is not None:
        ax.plot(x_vals, target_func(x_vals), ":", color="k", label="True")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(title + rf" $n_{{epochs}}={n_epochs}$")
    plt.legend()
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.set_ylabel(variable_label, rotation=45, fontsize="large")
    plt.tight_layout()

    if savePlots:
        plt.savefig(figsPath / f"{saveName}.pdf")
    if showPlots:
        plt.show()
    plt.close(fig)


def PlotErrorPerVariable(
    error_vals: list[np.ndarray],
    variable_arr: np.ndarray,
    x_label: str = "epoch",
    error_label: str = "MSE",
    variable_label: str = r"$\eta$",
    variable_type: str = "log",
    error_type: str = "linear",
    title: str = "Error per epoch",
    colormap: str = "viridis",
    savePlots: bool = False,
    showPlots: bool = True,
    figsPath: Path = None,
    saveName: str = None,
) -> None:
    r"""
    Plots training error for different values of a variable parameter.

    Args:
        error_vals (List[np.ndarray]):
            List of arrays of error values to plot, one for each value of the variable parameter.
        variable_arr (np.ndarray):
            Array of values of the variable parameter.
        x_label (str, optional):
            Label for the x-axis. Defaults to "epoch".
        error_label (str, optional):
            Label for the y-axis. Defaults to "MSE".
        variable_label (str, optional):
            Label for the variable parameter. Defaults to r"$\eta$".
        title (str, optional):
            Title for the plot. Defaults to "Error per epoch".
        savePlots (bool, optional):
            Whether to save the plot as a PNG file. Defaults to False.
        showPlots (bool, optional):
            Whether to display the plot. Defaults to True.
        figsPath (Path, optional):
            Path to the directory where the plot should be saved. Defaults to None.
        saveName (str, optional):
            Name of the file to save the plot as. Defaults to None.

    Returns:
        None
    """
    fig, ax = plt.subplots()

    cmap, norm, sm = setColors(
        variable_arr, cmap_name=colormap, norm_type=variable_type
    )

    ax.set_xlim(0, error_vals.shape[1])
    ax.set_ylim(np.nanmin(error_vals), np.nanmax(error_vals))

    for i, error in enumerate(error_vals):
        ax.plot(error, color=cmap(norm(variable_arr[i])))

    ax.set_xlabel(x_label)
    ax.set_ylabel(error_label)
    ax.set_yscale(error_type)
    plt.title(f"{title} ({variable_label})")
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.set_ylabel(variable_label, rotation=45, fontsize="large")
    plt.tight_layout()

    if savePlots:
        plt.savefig(figsPath / f"{saveName}.pdf")
    if showPlots:
        plt.show()
    plt.close(fig)


def plotThetas(
    theta_arr: np.ndarray,
    variable_arr: np.ndarray,
    variable_type: str = "log",
    true_theta: np.ndarray = None,
    variable_label: str = r"$\eta$",
    title: str = r"Values for $\theta$ for different values of ",
    colormap: str = "viridis",
    savePlots: bool = False,
    showPlots: bool = True,
    figsPath: Path = None,
    saveName: str = None,
) -> None:
    r"""
    Plots the values of theta for different values of eta.

    Args:
        theta_arr (np.ndarray):
            Array of theta values for different values.
        variable_arr (np.ndarray):
            Array of x values.
        variable_type (str, optional):
            Scale of the x-axis. Defaults to "log".
        true_theta (np.ndarray, optional):
            Array of true theta values. Defaults to None.
        variable_label (str, optional):
            Label for the x-axis. Defaults to r"$eta$".
        title (str, optional):
            Title of the plot. Defaults to r"Values for $\theta$ for different values of $\eta$".
        colormap (str, optional):
            Name of the colormap to use. Defaults to "viridis".
        savePlots (bool, optional):
            Whether to save the plot. Defaults to False.
        showPlots (bool, optional):
            Whether to show the plot. Defaults to True.
        figsPath (Path, optional):
            Path to the directory where the plot will be saved. Defaults to None.
        saveName (str, optional):
            Name of the file to save the plot as. Defaults to None.
    """
    tmp_arr = np.linspace(1, 2, theta_arr.shape[1])

    cmap, norm, sm = setColors(tmp_arr, cmap_name=colormap, norm_type=variable_type)

    for i in range(theta_arr.shape[1]):
        plt.plot(variable_arr, theta_arr[:, i], color=cmap(norm(tmp_arr[i])))

    if true_theta is not None:
        for i in range(theta_arr.shape[1]):
            plt.axhline(
                true_theta[i],
                color=cmap(norm(tmp_arr[i])),
                linestyle=":",
                label=rf"$\theta_{{{i}}}$",
            )
        plt.legend()

    title = title + " " + variable_label
    plt.xscale(variable_type)
    plt.xlabel(variable_label)
    plt.ylabel(r"$\theta$")
    plt.title(title)
    if savePlots:
        plt.savefig(figsPath / f"{saveName}.pdf")
    if showPlots:
        plt.show()
    plt.close()


def plotHeatmap(
    df: DataFrame,
    title: str = "Error",
    x_label: str = r"$\eta$",
    y_label: str = r"$\rho$",
    colormap: str = "viridis",
    savePlots: bool = False,
    showPlots: bool = True,
    figsPath: Path = None,
    saveName: str = None,
    annot: bool = False,
) -> None:
    r"""Plot a heatmap of the input DataFrame.

    Args:
        df (DataFrame): The DataFrame to plot.
        title (str, optional): The title of the plot. Defaults to "Error".
        x_label (str, optional): The label for the x-axis. Defaults to r"$\eta$".
        y_label (str, optional): The label for the y-axis. Defaults to r"$\rho$".
        colormap (str, optional): The name of the colormap to use. Defaults to "viridis".
        savePlots (bool, optional): Whether to save the plot. Defaults to False.
        showPlots (bool, optional): Whether to show the plot. Defaults to True.
        figsPath (Path, optional): Path to the directory where the plot will be saved. Defaults to None.
        saveName (str, optional): Name of the file to save the plot as. Defaults to None.
        annot (bool, optional): Whether to annotate the heatmap. Defaults to False.
    """
    fig, ax = plt.subplots()
    sns.heatmap(df, cmap=colormap, ax=ax, annot=annot, fmt=".3f")
    plt.title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.tight_layout()
    if savePlots:
        plt.savefig(figsPath / f"{saveName}.pdf", bbox_inches="tight")
    if showPlots:
        plt.show()
    plt.close(fig)


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


def plot_Franke_input(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    savePlots: bool = False,
    showPlots: bool = True,
    figsPath: Path = None,
    saveName: str = None,
) -> None:
    """Plot the surface of Franke's function.

    Args:
        x (np.ndarray): The x-values.
        y (np.ndarray): The y-values.
        z (np.ndarray): The z-values.
        savePlots (bool, optional): Whether to save the plot. Defaults to False.
        showPlots (bool, optional): Whether to show the plot. Defaults to True.
        figsPath (Path, optional): Path to the directory where the plot will be saved. Defaults to None.
        saveName (str, optional): Name of the file to save the plot as. Defaults to None.
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(
        x,
        y,
        z.reshape(x.shape),
        cmap="coolwarm",
        linewidth=0,
        antialiased=False,
    )
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    ax.set_title("Input data for Franke's function")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()

    if savePlots:
        plt.savefig(figsPath / f"{saveName}.pdf", bbox_inches="tight")
    if showPlots:
        plt.show()
    plt.close(fig)


def plot_Franke(
    FFNN: NeuralNet,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str = "Franke's function",
    savePlots: bool = False,
    showPlots: bool = True,
    figsPath: Path = None,
    saveName: str = None,
) -> None:
    """Plot the surface of Franke's function.

    Args:
        FFNN (NeuralNey): The neural network to use for prediction.
        x (np.ndarray): The x-values.
        y (np.ndarray): The y-values.
        z (np.ndarray): The z-values.
        savePlots (bool, optional): Whether to save the plot. Defaults to False.
        showPlots (bool, optional): Whether to show the plot. Defaults to True.
        figsPath (Path, optional): Path to the directory where the plot will be saved. Defaults to None.
        saveName (str, optional): Name of the file to save the plot as. Defaults to None.
    """
    xnew = np.linspace(0, 1, 100)
    ynew = np.linspace(0, 1, 100)
    xnew, ynew = np.meshgrid(xnew, ynew)
    znew = FFNN.predict(
        np.c_[xnew.ravel().reshape(-1, 1), ynew.ravel().reshape(-1, 1)]
    ).reshape(100, 100)

    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax.plot_surface(
        xnew,
        ynew,
        znew,
        cmap="coolwarm",
        linewidth=0,
        antialiased=False,
    )

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    ax.set_title("Predicted surface")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=10)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    surf = ax.plot_surface(
        xnew,
        ynew,
        np.abs(z.reshape(x.shape) - znew),
        cmap="coolwarm",
        cstride=1,
        rstride=1,
        linewidth=0,
        antialiased=False,
    )
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    ax.set_title("Absolute error")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=10)
    fig.suptitle(title)
    plt.tight_layout()

    if savePlots:
        plt.savefig(figsPath / f"{saveName}_surface.pdf", bbox_inches="tight")
    if showPlots:
        plt.show()
    plt.close(fig)


def plot_validation_train(
    scores: dict[str, np.ndarray[float]],
    title: str = "Error",
    savePlots: bool = False,
    showPlots: bool = True,
    figsPath: Path = None,
    saveName: str = None,
) -> None:
    """Plot the validation and training error.

    Args:
        scores (dict[str, np.ndarray[float]]): The scores dictionary.
        title (str, optional): The title of the plot. Defaults to "Error".
        savePlots (bool, optional): Whether to save the plot. Defaults to False.
        showPlots (bool, optional): Whether to show the plot. Defaults to True.
        figsPath (Path, optional): Path to the directory where the plot will be saved. Defaults to None.
        saveName (str, optional): Name of the file to save the plot as. Defaults to None.
    """
    fig, ax = plt.subplots()
    train_error = scores["train_errors"]
    ax.plot(np.log10(train_error), label="Train")
    validation_error = scores["validation_errors"]
    ax.plot(np.log10(validation_error), label="Validation")

    ax.set_ylabel(r"$\log_{10}$ Cost")
    ax.set_xlabel(r"$n_{epochs}$")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if savePlots:
        plt.savefig(figsPath / f"{saveName}_error_val.pdf", bbox_inches="tight")
    if showPlots:
        plt.show()
    plt.close(fig)
