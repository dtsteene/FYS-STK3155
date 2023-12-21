import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path


def plotBeta(
    betas: list[np.array],
    title: str,
    methodname: str = "OLS",
    savePlots: bool = False,
    showPlots: bool = True,
    figsPath: Path = Path(".").parent,
) -> None:
    """
    Plots the beta values of a linear regression model for different polynomial degrees.

    inputs:
        betas (list[np.array]):
            A list of beta values for different polynomial degrees.
        title (str):
            A string containing the title for the plot.
        methodname (str):
            A string containing the name of the method used to calculate beta values.
    """
    for i, beta in enumerate(betas[:5]):
        for param in beta:
            plt.scatter(i + 1, param, c="r", alpha=0.5)

    plt.title(f"{title} for {methodname}")
    plt.xticks([dim + 1 for dim in range(5)])
    plt.xlabel("Polynomial degree")
    plt.ylabel(r"$\beta_i$ value")

    tmp = []
    for beta in betas[:5]:
        tmp += list(beta.ravel())

    maxBeta = max(abs(min(tmp)), abs(max(tmp))) * 1.2

    plt.ylim((-maxBeta, maxBeta))

    if savePlots:
        plt.savefig(figsPath / f"{methodname}_{5}_betas.png", dpi=300)
    if showPlots:
        plt.show()
    plt.close()
