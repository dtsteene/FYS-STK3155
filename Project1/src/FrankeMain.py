# make all plots
import numpy as np

from pathlib import Path
from OLSRegression import (
    OLS_train_test,
    plot_Bias_VS_Variance,
    bootstrap_vs_cross_val,
)
from Models import Ridge, Lasso
from Data import FrankeData
from RegularizedRegression import (
    heatmap_no_resampling,
    heatmap_bootstrap,
    heatmap_sklearn_cross_val,
    heatmap_HomeMade_cross_val,
)
import sklearn.linear_model as sklm

np.random.seed(32019)
maxDim = 13
lmbds = np.logspace(-7, 3, 11)
figsPath = Path(__file__).parent.parent / "figures" / "Franke"

data = FrankeData(40, 0.2, maxDim, savePlots=True, showPlots=False, figsPath=figsPath)


# make franke plot
def Franke():
    data.plotSurface()


# OLS
def OLSAnalysis() -> None:
    "Run all the plots for Ordinary Least Squares"
    OLS_train_test(
        data, savePlots=True, showPlots=False, figsPath=figsPath, maxDim=maxDim
    )
    # BVData = FrankeData(
    #     20, 0.2, maxDim=25, savePlots=True, showPlots=False, figsPath=figsPath
    # )
    # plot_Bias_VS_Variance(
    #     BVData,
    #     maxDim=13,
    #     showPlots=False,
    #     savePlots=True,
    #     figsPath=figsPath,
    #     title="Few points Bias Variance Tradeoff",
    # )
    plot_Bias_VS_Variance(
        data, maxDim=15, savePlots=True, showPlots=False, figsPath=figsPath
    )
    bootstrap_vs_cross_val(
        data,
        maxDim=15,
        savePlots=True,
        showPlots=False,
        figsPath=figsPath,
        n_bootstraps=100,
    )


# Same analysis for lasso and Ridge as OLS
def RidgeAnalysis() -> None:
    "Run all the plots for Ridge"
    heatmap_no_resampling(
        data,
        maxDim=15,
        lmbds=lmbds,
        model=Ridge(),
        savePlots=True,
        showPlots=False,
        title="MSE Ridge no resampling",
        figsPath=figsPath,
    )
    heatmap_sklearn_cross_val(
        data,
        maxDim=maxDim,
        lmbds=lmbds,
        model=sklm.Ridge(),
        title="MSE Ridge CV from Scikit-learn",
        savePlots=True,
        showPlots=False,
        figsPath=figsPath,
    )
    heatmap_HomeMade_cross_val(
        data,
        maxDim=maxDim,
        lmbds=lmbds,
        model=Ridge(),
        title="MSE Ridge CV",
        savePlots=True,
        showPlots=False,
        figsPath=figsPath,
    )
    heatmap_bootstrap(
        data,
        maxDim=maxDim,
        lmbds=lmbds,
        model=Ridge(),
        title="MSE Ridge Bootstrap",
        savePlots=True,
        showPlots=False,
        figsPath=figsPath,
        n_bootstraps=100,
    )


def LassoAnalysis() -> None:
    "Run all the plots for Lasso"
    lmbds = np.logspace(-4, 2, 11)
    data = FrankeData(
        20, 0.2, maxDim, savePlots=True, showPlots=False, figsPath=figsPath
    )

    heatmap_no_resampling(
        data,
        model=Lasso(),
        maxDim=15,
        lmbds=lmbds,
        title="MSE Lasso no resampling",
        savePlots=True,
        showPlots=False,
        figsPath=figsPath,
    )
    heatmap_sklearn_cross_val(
        data,
        maxDim=maxDim,
        lmbds=lmbds,
        model=sklm.Lasso(),
        title="MSE Lasso CV from Scikit-learn",
        savePlots=True,
        showPlots=False,
        figsPath=figsPath,
    )
    heatmap_HomeMade_cross_val(
        data,
        maxDim=maxDim,
        lmbds=lmbds,
        model=Lasso(),
        title="MSE Lasso CV",
        savePlots=True,
        showPlots=False,
        figsPath=figsPath,
        kfolds=5,
    )
    heatmap_bootstrap(
        data,
        maxDim=maxDim,
        lmbds=lmbds,
        model=Lasso(),
        title="MSE Lasso Bootstrap",
        savePlots=True,
        showPlots=False,
        figsPath=figsPath,
        n_bootstraps=10,
    )


if __name__ == "__main__":
    Franke()
    OLSAnalysis()
    RidgeAnalysis()
    LassoAnalysis()
