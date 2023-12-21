from Models import OLS, Model
from Data import Data
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from tqdm import tqdm
from resampling import bootstrap_degrees, kfold_score_degrees
from plotBetas import plotBeta
from metrics import MSE, R2Score

# from globals import *


def plotScores(
    data: Data,
    MSEs_train: np.array,
    MSEs_test: np.array,
    R2s: np.array,
    methodname: str = "OLS",
    savePlots: bool = False,
    showPlots: bool = True,
    figsPath: Path = Path(".").parent,
    maxDim: int = 15,
) -> None:
    """Plots MSE_train, MSE_test, and R2 values as a function of polynomial degree.

    inputs:
        MSE_train (np.array):
            MSE of traning data and model prediction

        MSE_test (np.array):
            MSE of test data and model prediction

        R2 (np.array):
            R-squared score of test data and model prediction

        methodname (str):
            Name of method used to generate values

        savePlots (bool):
            Whether to save plots

        showPlots (bool):
            Whether to show plots

        figsPath (Path):
            Where to save files to
    """
    xVals = [i + 1 for i in range(maxDim)]

    color = "tab:red"
    plt.xlabel("Polynomial degree")
    plt.xticks(xVals)
    plt.ylabel("MSE score")
    plt.plot(xVals, MSEs_train, label="MSE train", color="r")
    plt.plot(xVals, MSEs_test, label="MSE test", color="g")

    minTestMSE = np.argmin(MSEs_test)
    plt.scatter(
        minTestMSE + 1,
        MSEs_test[minTestMSE],
        marker="x",
        label="Minimum test error",
    )
    plt.title(f"Mean squared error for {methodname}")
    plt.legend()

    if savePlots:
        plt.savefig(figsPath / f"{methodname}_{maxDim}_MSE.png", dpi=300)
    if showPlots:
        plt.show()
    plt.close()

    color = "tab:blue"
    plt.xlabel("Polynomial degree")
    plt.xticks(xVals)
    plt.ylabel("$R^2$")
    plt.plot(xVals, R2s, label="$R^2$ score", color=color)

    maxR2 = np.argmax(R2s)
    plt.scatter(maxR2 + 1, R2s[maxR2], marker="x", label="Maximum $R^2$")

    plt.legend()
    plt.title(f"$R^2$ Scores by polynomial degree for {methodname}")

    if savePlots:
        plt.savefig(figsPath / f"{methodname}_{maxDim}_R2.png", dpi=300)
    if showPlots:
        plt.show()
    plt.close()


def OLS_train_test(data: Data, maxDim: int, **kwargs) -> None:
    """Validation set for OLS.

    inputs:
        data (Data): Data to plot with
        **kwargs: savePlot, showPlots, figsPath, ...
    """
    betas = []
    MSETrain = np.zeros(maxDim)
    MSETest = np.zeros(maxDim)
    R2Scores = np.zeros(maxDim)

    pbar = tqdm(total=maxDim, desc="OLS MSEs")
    for dim in range(maxDim):
        model = OLS()
        X_train = model.create_X(data.x_train, data.y_train, dim)
        X_test = model.create_X(data.x_test, data.y_test, dim)

        beta = model.fit(X_train, data.z_train)
        betas.append(beta)

        z_tilde = X_train @ beta
        z_pred = X_test @ beta

        MSETrain[dim] = MSE(data.z_train, z_tilde)
        MSETest[dim] = MSE(data.z_test, z_pred)
        R2Scores[dim] = R2Score(data.z_test, z_pred)
        pbar.update(1)

    plotBeta(betas, r"$\beta$", "OLS", **kwargs)
    plotScores(data, MSETrain, MSETest, R2Scores, "OLS", maxDim=maxDim, **kwargs)


def plot_Bias_VS_Variance(
    data: Data,
    maxDim: int = 5,
    title: str = None,
    savePlots: bool = False,
    showPlots: bool = False,
    figsPath: Path = Path(".").parent,
    n_bootstraps: int = 100,
) -> None:
    """Plots variance, error(MSE), and bias using bootstrapping as resampling technique.

    inputs:
        data (ndarray):
            A one-dimensional numpy array containing the data set to be analyzed.

        maxDim (int):
            Maximum number of dimension

        title (str):
            Title of plot

        savePlots (bool):
            Whether to save the plots

        showPlots (bool):
            Whether to show the plots

        figsPath (Path):
            Path to save figures to

        n_bootstraps (int):
            Optional, default is 100. Number of random samples to generate using bootstrapping
    """
    error, bias, variance = bootstrap_degrees(data, n_bootstraps, OLS(), maxDim=maxDim)
    polyDegrees = range(1, maxDim + 1)

    if title is None:
        title = "Bias Variance Tradeoff OLS"

    plt.plot(polyDegrees, error, label="Error")
    plt.plot(polyDegrees, bias, label="Bias")
    plt.plot(polyDegrees, variance, label="Variance")
    plt.xlabel("Polynomial degree")
    plt.xticks(polyDegrees)
    plt.title(title)
    plt.legend()
    if savePlots:
        plt.savefig(figsPath / f"{'_'.join(title.split())}_{maxDim}.png", dpi=300)
    if showPlots:
        plt.show()
    plt.close()


def bootstrap_vs_cross_val(
    data,
    maxDim: int = 5,
    savePlots: bool = False,
    showPlots: bool = True,
    figsPath: Path = None,
    kfolds: int = 10,
    n_bootstraps: int = 100,
    model: Model = OLS(),
) -> None:
    """Compare bootstrap and Cross validation scores for OLS

    inputs:
        data (ndarray):
            A one-dimensional numpy array containing the data set to be analyzed.

        maxDim (int):
            Maximum number of dimension

        title (str):
            Title of plot

        savePlots (bool):
            Whether to save the plots

        showPlots (bool):
            Whether to show the plots

        figsPath (Path):
            Path to save figures to

        kfolds (int):
            Number of folds to create

        n_bootstraps (int):
            Optional, default is 1000. Number of random samples to generate using bootstrapping
    """
    polyDegrees = range(1, maxDim + 1)
    error_kfold, variance_kfold = kfold_score_degrees(
        data,
        kfolds=kfolds,
        maxDim=maxDim,
        model=model,
    )
    error_boot, bias_boot, variance_boot = bootstrap_degrees(
        data,
        n_bootstraps=n_bootstraps,
        maxDim=maxDim,
        model=model,
    )

    modelname = model.__class__.__name__
    plt.plot(polyDegrees, error_kfold, "b", label="Kfold CV Error")
    plt.plot(polyDegrees, error_boot, "r--", label="Bootstrap Error")
    plt.title(f"Bootstrap vs CV for {modelname}")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Error")
    plt.legend()
    if savePlots:
        plt.savefig(figsPath / f"Bias_vs_Var_{modelname}_{maxDim}.png", dpi=300)
    if showPlots:
        plt.show()
    plt.close()
