import numpy as onp
import jax.numpy as np
from Schedules import (
    Scheduler,
    Constant,
    Momentum,
    Adagrad,
    AdagradMomentum,
    Adam,
    RMS_prop,
    TimeDecay,
)
from Gradients import Gradients
from plotutils import (
    plotThetas,
    PlotPredictionPerVariable,
    PlotErrorPerVariable,
    plotHeatmap,
)
from typing import Callable
import pandas as pd
from sklearn.metrics import mean_squared_error
from utils import assign, assign_row
from pathlib import Path


class GradAnalysis:
    def __init__(
        self,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        n_epochs: int = 500,
        seed: int = None,
        cost_func: Callable = None,
        derivative_func: Callable = None,
        base_theta: np.ndarray = None,
        target_func: Callable = None,
        true_theta: np.ndarray = None,
        base_x: np.ndarray = None,
        lmbda: float = None,
        showPlots: bool = True,
        savePlots: bool = False,
        figspath: Path = None,
        polynomial: bool = True,
    ) -> None:
        """Base class for analyzing gradient descent methods.

        Args:
            x_vals (np.ndarray): The input data.
            y_vals (np.ndarray): The output data.
            n_epochs (int, optional): The number of epochs to run. Defaults to 500.
            seed (int, optional): The random seed to use. Defaults to None.
            cost_func (Callable, optional): The cost function to use. Defaults to None.
            derivative_func (Callable, optional): The derivative of the cost function to use. Defaults to None.
            base_theta (np.ndarray, optional): The initial model parameters. Defaults to None.
            target_func (Callable, optional): The target function to use. Defaults to None.
            true_theta (np.ndarray, optional): The true model parameters. Defaults to None.
            base_x (np.ndarray, optional): The input data to use for plotting. Defaults to None.
            lmbda (float, optional): The regularization parameter. Defaults to None.
            showPlots (bool, optional): Whether to show the plots. Defaults to True.
            savePlots (bool, optional): Whether to save the plots. Defaults to False.
            figspath (Path, optional): The path to save the plots. Defaults to None.
            polynomial (bool, optional): Whether to use polynomial features. Defaults to True.
        """
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.n_epochs = n_epochs
        self.n_points = len(x_vals)
        self.seed = seed
        self.cost_func = cost_func
        self.derivative_func = derivative_func
        self.lmbda = lmbda
        self.polynomial = polynomial

        self.savePlots = savePlots
        self.showPlots = showPlots
        self.figspath = figspath

        if self.seed is not None:
            onp.random.seed(self.seed)

        if base_theta is None:
            self.base_theta = onp.random.randn(3, 1)
        else:
            self.base_theta = base_theta.reshape(-1, 1)

        self.target_func = target_func
        self.true_theta = true_theta

        self.base_x = base_x
        if self.base_x is None:
            self.base_x = np.linspace(-2, 2, 100)

    def error_and_theta_vals_gd(
        self, schedulers: list[Scheduler], dim: int = 2
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate the model parameters and errors for a list of schedulers.

        Args:
            schedulers (list[Scheduler]): The schedulers to use.
            dim (int, optional): The degree of the polynomial to use. Defaults to 2.

        Returns:
            theta_arr (np.ndarray): The final model parameters for each scheduler.
            error_Arr (np.ndarray): The error per epoch for each scheduler.
        """
        theta_arr = np.zeros((len(schedulers), dim + 1))
        error_arr = np.zeros((len(schedulers), self.n_epochs))

        for i, schedule in enumerate(schedulers):
            Gradient = Gradients(
                self.n_points,
                self.x_vals,
                self.y_vals,
                self.cost_func,
                self.derivative_func,
                schedule,
                lmbda=self.lmbda,
                polynomial=self.polynomial,
            )

            theta_arr = assign_row(
                theta_arr,
                i,
                Gradient.GradientDescent(self.base_theta, self.n_epochs).ravel(),
            )

            error_arr = assign_row(error_arr, i, Gradient.errors)

        return theta_arr, error_arr

    def error_per_variables(
        self, schedules: list[list[Scheduler]], dim: int = 2
    ) -> np.ndarray:
        """Generate the error for a nested list of schedulers, varying to parameters.

        Args:
            schedules (list[list[Scheduler]]): The schedulers to use.
            dim (int, optional): The degree of the polynomial to use. Defaults to 2.

        Returns:
            error_arr (np.ndarray): The error per epoch for each scheduler.
        """
        error_arr = np.zeros((len(schedules), len(schedules[0])))
        ynew = self.target_func(self.base_x)

        Gradient = Gradients(
            self.n_points,
            self.x_vals,
            self.y_vals,
            self.cost_func,
            self.derivative_func,
            Constant(0.1),
            lmbda=self.lmbda,
            polynomial=self.polynomial,
        )
        for i, row in enumerate(schedules):
            for j, schedule in enumerate(row):
                Gradient.scheduler = schedule
                theta = Gradient.GradientDescent(self.base_theta, self.n_epochs)
                ypred = Gradient.predict(self.base_x, theta, dim)
                error_arr = assign(error_arr, (i, j), mean_squared_error(ynew, ypred))

        return error_arr

    def pred_per_theta(
        self, x: np.ndarray, theta_arr: np.ndarray, dim: int = 2
    ) -> np.ndarray:
        """Generate predictions for a list of model parameters.

        Args:
            x (np.ndarray): The input data.
            theta_arr (np.ndarray): The model parameters.
            dim (int, optional): The degree of the polynomial to use. Defaults to 2.

        Returns:
            pred_arr (np.ndarray): The predictions for each set of model parameters.
        """
        pred_arr = np.zeros((len(theta_arr), self.n_points))

        DummyGrad = Gradients(
            self.n_points,
            self.x_vals,
            self.y_vals,
            self.cost_func,
            self.derivative_func,
            Constant(1),
            lmbda=self.lmbda,
            polynomial=self.polynomial,
        )
        for i, theta in enumerate(theta_arr):
            pred_arr = assign_row(pred_arr, i, DummyGrad.predict(x, theta, dim))

        return pred_arr

    def error_per_minibatch(
        self,
        schedule: Scheduler,
        minibatches: np.ndarray,
        n_epochs: int = 150,
        dim: int = 2,
    ) -> tuple:
        """Generate the error and model parameters for a list of minibatch sizes.

        Args:
            schedule (Scheduler): The scheduler to use.
            minibatches (np.ndarray): The minibatch sizes to use.
            n_epochs (int, optional): The number of epochs to run. Defaults to 150.
            dim (int, optional): The degree of the polynomial to use. Defaults to 2.

        Returns:
            error_arr (np.ndarray): The error per epoch for each minibatch size.
            theta_arr (np.ndarray): The final model parameters for each minibatch size.
        """
        error_arr = np.zeros((len(minibatches), n_epochs))
        theta_arr = np.zeros((len(minibatches), dim + 1))
        for i, batch_size in enumerate(minibatches):
            Gradient = Gradients(
                self.n_points,
                self.x_vals,
                self.y_vals,
                self.cost_func,
                self.derivative_func,
                schedule,
                lmbda=self.lmbda,
                polynomial=self.polynomial,
            )
            theta_arr = assign_row(
                theta_arr,
                i,
                Gradient.StochasticGradientDescent(
                    self.base_theta, n_epochs, batch_size
                ).ravel(),
            )

            error_arr = assign_row(error_arr, i, Gradient.errors)

        return error_arr, theta_arr

    def constant_analysis(self, eta_vals: np.ndarray, dim: int = 2) -> None:
        """Analyze the constant scheduler, generating multiple plots.

        Args:
            eta_vals (np.ndarray): The learning rates to use.
            dim (int, optional): The degree of the polynomial to use. Defaults to 2.
        """
        schedulers = [Constant(eta) for eta in eta_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            eta_vals,
            title="Error per epoch (Constant)",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="constant_error",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            eta_vals,
            title="Predicted polynomials (Constant)",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="constant_prediction",
        )
        plotThetas(
            theta_arr,
            eta_vals,
            title="Model parameters (Constant)",
            true_theta=self.true_theta,
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="constant_thetas",
        )

    def momentum_analysis(
        self,
        eta_vals: np.ndarray,
        rho_vals: np.ndarray,
        dim: int = 2,
    ) -> None:
        """Analyze the momentum scheduler, generating multiple plots.

        Args:
            eta_vals (np.ndarray): The learning rates to use.
            rho_vals (np.ndarray): The momentum parameters to use.
            dim (int, optional): The degree of the polynomial to use. Defaults to 2.
        """
        schedulers = [Momentum(eta, 0.9) for eta in eta_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            eta_vals,
            title=r"Error per epoch (Momentum) $\rho=0.9$",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="momentum_error_eta",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            eta_vals,
            title=r"Predicted polynomials (Momentum) $\rho=0.9$",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="momentum_prediction_eta",
        )
        plotThetas(
            theta_arr,
            eta_vals,
            title=r"Model parameters (Momentum) $\rho=0.9$",
            true_theta=self.true_theta,
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="momentum_thetas_eta",
        )

        schedulers = [Momentum(0.001, rho) for rho in rho_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            rho_vals,
            title=r"Error per epoch (Momentum) $\eta=0.001$",
            variable_label=r"$\rho$",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="momentum_error_rho",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            rho_vals,
            title=r"Predicted polynomials (Momentum) $\eta=0.001$",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
            variable_label=r"$\rho$",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="momentum_prediction_rho",
        )
        plotThetas(
            theta_arr,
            rho_vals,
            title=r"Model parameters (Momentum) $\eta=0.001$",
            true_theta=self.true_theta,
            variable_label=r"$\rho$",
            variable_type="linear",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="momentum_thetas_rho",
        )

        n_rho = 75
        n_eta = 75
        heat_rho = np.arctan(np.linspace(0, 10, n_rho))
        heat_rho = heat_rho / np.max(heat_rho)
        heat_eta = np.logspace(-7, -1, n_eta)

        schedulers = [[Momentum(eta, rho) for eta in heat_eta] for rho in heat_rho]
        error_arr = self.error_per_variables(schedulers, dim)

        df = pd.DataFrame(
            error_arr,
            index=[f"{rho:.2f}" for rho in heat_rho],
            columns=[f"{eta:.2e}" for eta in heat_eta],
        )
        plotHeatmap(
            df,
            title=f"Error after {self.n_epochs} epochs (Momentum)",
            x_label=r"$\eta$",
            y_label=r"$\rho$",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="momentum_heatmap_eta_rho",
        )

    def adagrad_analysis(
        self,
        eta_vals: np.ndarray,
        dim: int = 2,
    ) -> None:
        """Analyze the adagrad scheduler, generating multiple plots.

        Args:
            eta_vals (np.ndarray): The learning rates to use.
            dim (int, optional): The degree of the polynomial to use. Defaults to 2.
        """
        schedulers = [Adagrad(eta) for eta in eta_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            eta_vals,
            title=r"Error per epoch (Adagrad)",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adagrad_error",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            eta_vals,
            title=r"Predicted polynomials (Adagrad)",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adagrad_prediction",
        )
        plotThetas(
            theta_arr,
            eta_vals,
            title=r"Model parameters (Adagrad)",
            true_theta=self.true_theta,
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adagrad_thetas",
        )

    def adagrad_momentum_analysis(
        self,
        eta_vals: np.ndarray,
        rho_vals: np.ndarray,
        dim: int = 2,
    ) -> None:
        """Analyze the adagrad momentum scheduler, generating multiple plots.

        Args:
            eta_vals (np.ndarray): The learning rates to use.
            rho_vals (np.ndarray): The momentum parameters to use.
            dim (int, optional): The degree of the polynomial to use. Defaults to 2.
        """
        schedulers = [AdagradMomentum(eta, 0.9) for eta in eta_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            eta_vals,
            title=r"Error per epoch (AdagradMomentum) $\rho=0.9$",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adagrad_momentum_error_eta",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            eta_vals,
            title=r"Predicted polynomials (AdagradMomentum) $\rho=0.9$",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adagrad_momentum_prediction_eta",
        )
        plotThetas(
            theta_arr,
            eta_vals,
            title=r"Model parameters (AdagradMomentum) $\rho=0.9$",
            true_theta=self.true_theta,
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adagrad_momentum_thetas_eta",
        )

        schedulers = [AdagradMomentum(0.1, rho) for rho in rho_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            rho_vals,
            title=r"Error per epoch (AdagradMomentum) $\eta=0.1$",
            variable_label=r"$\rho$",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adagrad_momentum_error_rho",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            rho_vals,
            title=r"Predicted polynomials (AdagradMomentum) $\eta=0.1$",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
            variable_label=r"$\rho$",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adagrad_momentum_prediction_rho",
        )
        plotThetas(
            theta_arr,
            rho_vals,
            title=r"Model parameters (AdagradMomentum) $\eta=0.1$",
            true_theta=self.true_theta,
            variable_label=r"$\rho$",
            variable_type="linear",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adagrad_momentum_thetas_rho",
        )

        n_rho = 75
        n_eta = 75
        heat_rho = np.arctan(np.linspace(0, 10, n_rho))
        # Scale rho to be between 0 and 1 (Want to remove the outliers just below 1)
        heat_rho = heat_rho * 2 / 3
        heat_eta = np.logspace(-3, 0, n_eta)

        schedulers = [
            [AdagradMomentum(eta, rho) for eta in heat_eta] for rho in heat_rho
        ]
        error_arr = self.error_per_variables(schedulers, dim)
        df = pd.DataFrame(
            error_arr,
            index=[f"{rho:.2f}" for rho in heat_rho],
            columns=[f"{eta:.2e}" for eta in heat_eta],
        )
        plotHeatmap(
            df,
            title=f"Error after {self.n_epochs} epochs (AdagradMomentum)",
            x_label=r"$\eta$",
            y_label=r"$\rho$",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adagrad_momentum_heatmap_eta_rho",
        )

    def adam_analysis(
        self,
        eta_vals: np.ndarray,
        rho_vals: np.ndarray,
        rho2_vals: np.ndarray,
        dim: int = 2,
    ) -> None:
        """Analyze the adam scheduler, generating multiple plots.

        Args:
            eta_vals (np.ndarray): The learning rates to use.
            rho_vals (np.ndarray): The rho_1 parameters to use.
            rho2_vals (np.ndarray): The rho_2 parameters to use.
            dim (int, optional): The degree of the polynomial to use. Defaults to 2.
        """
        schedulers = [Adam(eta, 0.9, 0.999) for eta in eta_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            eta_vals,
            title=r"Error per epoch (Adam) $\rho=0.9$, $\rho_2=0.999$",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adam_error_eta",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            eta_vals,
            title=r"Predicted polynomials (Adam) $\rho=0.9$, $\rho_2=0.999$",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adam_prediction_eta",
        )
        plotThetas(
            theta_arr,
            eta_vals,
            title=r"Model parameters (Adam) $\rho=0.9$, $\rho_2=0.999$",
            true_theta=self.true_theta,
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adam_thetas_eta",
        )

        schedulers = [Adam(0.1, rho, 0.999) for rho in rho_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            rho_vals,
            title=r"Error per epoch (Adam) $\eta=0.1$, $\rho_2=0.999$",
            variable_label=r"$\rho_1$",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adam_error_rho1",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            rho_vals,
            title=r"Predicted polynomials (Adam) $\eta=0.1$, $\rho_2=0.999$",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
            variable_label=r"$\rho_1$",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adam_prediction_rho1",
        )
        plotThetas(
            theta_arr,
            rho_vals,
            title=r"Model parameters (Adam) $\eta=0.1$, $\rho_2=0.999$",
            true_theta=self.true_theta,
            variable_label=r"$\rho_1$",
            variable_type="linear",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adam_thetas_rho1",
        )

        schedulers = [Adam(0.1, 0.9, rho2) for rho2 in rho2_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)

        PlotErrorPerVariable(
            error_arr,
            rho2_vals,
            title=r"Error per epoch (Adam) $\eta=0.1$, $\rho_1=0.9$",
            variable_label=r"$\rho_2$",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adam_error_rho2",
        )
        plotThetas(
            theta_arr,
            rho2_vals,
            title=r"Model parameters (Adam) $\eta=0.1$, $\rho_2=0.9$",
            true_theta=self.true_theta,
            variable_label=r"$\rho_2$",
            variable_type="linear",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adam_thetas_rho2",
        )

        n_rho = 75
        n_eta = 75
        heat_rho = np.arctan(np.linspace(0.1, 10, n_rho))
        heat_rho = heat_rho * 2 / 3
        heat_eta = np.logspace(-3, 0, n_eta)

        schedulers = [[Adam(eta, rho, 0.999) for eta in heat_eta] for rho in heat_rho]
        error_arr = self.error_per_variables(schedulers, dim)

        df = pd.DataFrame(
            error_arr,
            index=[f"{rho:.2f}" for rho in heat_rho],
            columns=[f"{eta:.2e}" for eta in heat_eta],
        )
        plotHeatmap(
            df,
            title=rf"Error after {self.n_epochs} epochs (Adam) ($\rho_2=0.999$)",
            x_label=r"$\eta$",
            y_label=r"$\rho_1$",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="adam_momentum_heatmap_eta_rho",
        )

    def rms_prop_analysis(
        self,
        eta_vals: np.ndarray,
        rho_vals: np.ndarray,
        dim: int = 2,
    ) -> None:
        """Analyze the RMSprop scheduler, generating multiple plots.

        Args:
            eta_vals (np.ndarray): The learning rates to use.
            rho_vals (np.ndarray): The rho parameters to use.
            dim (int, optional): The degree of the polynomial to use. Defaults to 2.
        """
        schedulers = [RMS_prop(eta, 0.9) for eta in eta_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            eta_vals,
            title=r"Error per epoch (RMS_prop) $\rho=0.9$",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="rms_prop_error_eta",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            eta_vals,
            title=r"Predicted polynomials (RMS_prop) $\rho=0.9$",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="rms_prop_prediction_eta",
        )
        plotThetas(
            theta_arr,
            eta_vals,
            title=r"Model parameters (RMS_prop) $\rho=0.9$",
            true_theta=self.true_theta,
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="rms_prop_thetas_eta",
        )

        schedulers = [RMS_prop(0.01, rho) for rho in rho_vals]
        theta_arr, error_arr = self.error_and_theta_vals_gd(schedulers, dim)
        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            rho_vals,
            title=r"Error per epoch (RMS_prop) $\eta=0.01$",
            variable_label=r"$\rho$",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="rms_prop_error_rho",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            rho_vals,
            title=r"Predicted polynomials (RMS_prop) $\eta=0.01$",
            n_epochs=self.n_epochs,
            target_func=self.target_func,
            variable_label=r"$\rho$",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="rms_prop_prediction_rho",
        )
        plotThetas(
            theta_arr,
            rho_vals,
            title=r"Model parameters (RMS_prop) $\eta=0.01$",
            true_theta=self.true_theta,
            variable_label=r"$\rho$",
            variable_type="linear",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="rms_prop_thetas_rho",
        )

    def minibatch_analysis(self, dim: int = 2) -> None:
        """Analyze the minibatch sizes, generating multiple plots.

        Runs over the different schedulers, may take a while.

        Args:
            dim (int, optional): The degree of the polynomial to use. Defaults to 2.
        """
        # Flip so that the "better" args are plotted on top
        minibatches = np.flip(np.arange(1, 101))

        error_arr = np.zeros((len(minibatches), 150))
        theta_arr = np.zeros((len(minibatches), dim + 1))
        for i, batch_size in enumerate(minibatches):
            schedule = TimeDecay(1, 10, batch_size)
            Gradient = Gradients(
                self.n_points,
                self.x_vals,
                self.y_vals,
                self.cost_func,
                self.derivative_func,
                schedule,
                lmbda=self.lmbda,
                polynomial=self.polynomial,
            )
            theta_arr = assign_row(
                theta_arr,
                i,
                Gradient.StochasticGradientDescent(
                    self.base_theta, 150, batch_size
                ).ravel(),
            )

            error_arr = assign_row(error_arr, i, Gradient.errors)

        pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

        PlotErrorPerVariable(
            error_arr,
            minibatches,
            title=r"Error per epoch (TimeDecay) $t_0 = 1$, $t_1 = 10$",
            variable_label="Minibatch size",
            variable_type="linear",
            colormap="viridis_r",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="time_decay_error_minibatch",
        )
        plotThetas(
            theta_arr,
            minibatches,
            title=r"Model parameters (TimeDecay) $t_0 = 1$, $t_1 = 10$",
            variable_label="Minibatch size",
            variable_type="linear",
            true_theta=self.true_theta,
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="time_decay_thetas_minibatch",
        )
        PlotPredictionPerVariable(
            self.base_x,
            pred_arr,
            minibatches,
            title=r"Predicted polynomials (TimeDecay) $t_0 = 1$, $t_1 = 10$",
            n_epochs=150,
            target_func=self.target_func,
            variable_label="Minibatch size",
            variable_type="linear",
            colormap="viridis_r",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="time_decay_prediction_minibatch",
        )

        schedules = [
            Constant(0.01),
            Momentum(0.001, 0.9),
            Adagrad(0.01),
            AdagradMomentum(0.01, 0.9),
            Adam(0.01, 0.9, 0.999),
            RMS_prop(0.01, 0.9),
        ]
        schedule_titles = [
            r"(Constant) $\eta=0.01$",
            r"(Momentum) $\eta=0.01$, $\rho=0.9$",
            r"(Adagrad) $\eta=0.01$",
            r"(AdagradMomentum) $\eta=0.01$, $\rho=0.9$",
            r"(Adam) $\eta=0.01$, $\rho_1=0.9$, $\rho_2=0.999$",
            r"(RMS_prop) $\eta=0.01$, $\rho=0.9$",
        ]
        saveNames = [
            "constant",
            "momentum",
            "adagrad",
            "adagrad_momentum",
            "adam",
            "rms_prop",
        ]
        epoch_size = [60, 60, 150, 60, 40, 60]
        for schedule, title, epoch, saveName in zip(
            schedules, schedule_titles, epoch_size, saveNames
        ):
            error_arr, theta_arr = self.error_per_minibatch(
                schedule, minibatches, epoch
            )
            pred_arr = self.pred_per_theta(self.base_x, theta_arr, dim)

            PlotErrorPerVariable(
                error_arr,
                minibatches,
                title=f"Error per epoch {title}",
                variable_label="Minibatch size",
                variable_type="linear",
                colormap="viridis_r",
                savePlots=self.savePlots,
                showPlots=self.showPlots,
                figsPath=self.figspath,
                saveName=f"{saveName}_error_minibatch",
            )
            plotThetas(
                theta_arr,
                minibatches,
                title=f"Model parameters {title}",
                variable_label="Minibatch size",
                variable_type="linear",
                true_theta=self.true_theta,
                savePlots=self.savePlots,
                showPlots=self.showPlots,
                figsPath=self.figspath,
                saveName=f"{saveName}_thetas_minibatch",
            )
            PlotPredictionPerVariable(
                self.base_x,
                pred_arr,
                minibatches,
                title=f"Predicted polynomials {title}",
                n_epochs=epoch,
                target_func=self.target_func,
                variable_label="Minibatch size",
                variable_type="linear",
                colormap="viridis_r",
                savePlots=self.savePlots,
                showPlots=self.showPlots,
                figsPath=self.figspath,
                saveName=f"{saveName}_prediction_minibatch",
            )

    def gd_main(self, dim: int = 2) -> None:
        """Run the gradient descent analysis, generating multiple plots.

        Runs through the different schedulers, may take a while.

        Args:
            dim (int, optional): The degree of the polynomial to use. Defaults to 2.
        """
        eta_num = 75
        eta_arr = np.logspace(-5, -1, eta_num)
        self.constant_analysis(eta_arr, dim)

        rho_num = 75
        rho_arr = np.linspace(1 / self.n_points, 1, rho_num)
        self.momentum_analysis(eta_arr, rho_arr, dim)

        # NOTE: Adagrad is less sensitive to the learning rate, so we can use larger values
        eta_arr = np.logspace(-3, 0, eta_num)
        self.adagrad_analysis(eta_arr, dim)

        eta_arr = np.logspace(-5, 0, eta_num)
        rho_arr = np.linspace(1 / self.n_points, 0.99, rho_num)
        self.adagrad_momentum_analysis(eta_arr, rho_arr, dim)

        rho2_num = 75
        rho2_arr = np.linspace(0.05, 0.999, rho2_num)
        self.adam_analysis(eta_arr, rho_arr, rho2_arr, dim)

        self.rms_prop_analysis(eta_arr, rho_arr, dim)

    def ridge_analysis(self, eta_arr: np.ndarray, lmbda_arr: np.ndarray) -> None:
        """Analyze the ridge regression, generating multiple plots.

        Args:
            eta_arr (np.ndarray): The learning rates to use.
            lmbda_arr (np.ndarray): The regularization parameters to use.
        """
        ynew = self.target_func(self.base_x)

        theta_arr = np.zeros((len(lmbda_arr), 3))
        Gradient = Gradients(
            self.n_points,
            self.x_vals,
            self.y_vals,
            self.cost_func,
            self.derivative_func,
            Constant(0.1),
            lmbda=0.1,
            polynomial=self.polynomial,
        )
        for i, lmbda in enumerate(lmbda_arr):
            Gradient.lmbda = lmbda
            theta_arr = assign_row(
                theta_arr,
                i,
                Gradient.GradientDescent(self.base_theta, self.n_epochs).ravel(),
            )

        plotThetas(
            theta_arr,
            lmbda_arr,
            true_theta=self.true_theta,
            title=r"$\theta$ for different values of $\lambda$ (Constant Ridge) $\eta=0.1$",
            variable_label=r"$\lambda$",
            savePlots=self.savePlots,
            showPlots=self.showPlots,
            figsPath=self.figspath,
            saveName="constant_ridge_thetas",
        )

        schedulers = [Constant, Momentum, Adagrad, AdagradMomentum, Adam, RMS_prop]
        params = [
            {},
            {"rho": 0.9},
            {},
            {"rho": 0.9},
            {"rho": 0.9, "rho2": 0.999},
            {"rho": 0.9},
        ]
        schedule_names = [
            r"Constant ($\eta=0.01$)",
            r"Momentum ($\eta=0.01$, $\rho=0.9$)",
            r"Adagrad ($\eta=0.01$)",
            r"AdagradMomentum ($\eta=0.01$, $\rho=0.9$)",
            r"Adam ($\eta=0.01$, $\rho=0.9$, $\rho_2=0.999$)",
            r"RMS_prop ($\eta=0.01$, $\rho=0.9$)",
        ]
        saveNames = [
            "constant",
            "momentum",
            "adagrad",
            "adagrad_momentum",
            "adam",
            "rms_prop",
        ]
        for schedule, param, schedule_name, saveName in zip(
            schedulers, params, schedule_names, saveNames
        ):
            error_arr = np.zeros((len(eta_arr), len(lmbda_arr)))

            for i, eta in enumerate(eta_arr):
                for j, lmbda in enumerate(lmbda_arr):
                    Gradient = Gradients(
                        self.n_points,
                        self.x_vals,
                        self.y_vals,
                        self.cost_func,
                        self.derivative_func,
                        schedule(eta=eta, **param),
                        lmbda=lmbda,
                    )
                    theta = Gradient.GradientDescent(self.base_theta, self.n_epochs)
                    ypred = Gradient.predict(self.base_x, theta, 2)
                    tmp = mean_squared_error(ynew, ypred)
                    error_arr = assign(error_arr, (i, j), tmp)

            df = pd.DataFrame(
                error_arr,
                index=[f"{eta:.2e}" for eta in eta_arr],
                columns=[f"{lmbda:.2e}" for lmbda in lmbda_arr],
            )
            plotHeatmap(
                df,
                title=f"Ridge {schedule_name}",
                x_label=r"$\lambda$",
                y_label=r"$\eta$",
                savePlots=self.savePlots,
                showPlots=self.showPlots,
                figsPath=self.figspath,
                saveName=f"{saveName}_ridge_heatmap_eta_lambda",
            )
