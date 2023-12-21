import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Rectangle


def create_heatmap(
    MSE_test,
    lmbds=np.linspace(-3, 5, 10),
    title: str = None,
    savePlots: bool = False,
    showPlots: bool = True,
    figsPath: Path = Path(".").parent,
) -> None:
    """Create heatmap of MSE for lambda vs degree.

    Creates a heatmap plot of the test set MSE given the results of the model for
    different polynomial degrees and lambda values.

    inputs:
        MSE_test (np.array):
            A 2D numpy array of shape (n_poly_degrees, n_lambdas) with test set MSE
            values for different polynomial degrees and lambda values.
        lmbds (np.array):
            A 1D numpy array of lambda values for regularization.
        title (str):
            A string containing the title for the heatmap plot.
    """

    # Define polynomial degrees and lambda values
    degrees = np.arange(1, len(MSE_test) + 1)

    fig, ax = plt.subplots(figsize=plt.figaspect(0.5))

    sns.heatmap(
        MSE_test,
        cmap="coolwarm",
        annot=True,
        fmt=".4f",
        cbar=True,
        annot_kws={"fontsize": 8},
        xticklabels=[f"{lmbda:.1f}" for lmbda in np.log10(lmbds)],
        yticklabels=degrees,
    )

    minX, minY = np.unravel_index(MSE_test.argmin(), MSE_test.shape)
    minIndex = (minY, minX)
    ax.add_patch(Rectangle(minIndex, 1, 1, fill=False, edgecolor="black", lw=3))

    ax.set_xlabel(r"$\log_{10} \lambda$")
    ax.set_ylabel("Polynomial Degree")

    # Set title
    if title is None:
        if savePlots:
            raise ValueError("Attempting to save file without setting title!")
        title = "MSE"

    ax.set_title(title, fontweight="bold", fontsize=20, pad=25)

    fig.tight_layout()

    if savePlots:
        plt.savefig(figsPath / f"Heatmap_{'_'.join(title.split())}.png", dpi=300)
    if showPlots:
        plt.show()
    plt.close()
