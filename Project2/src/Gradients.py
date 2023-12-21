import jax.numpy as np
import numpy as onp
from jax import grad, jit, lax
from Schedules import Scheduler, Constant
from typing import Callable
from CostFuncs import fast_OLS
from utils import assign, design, update_theta
from line_profiler import profile

np.random.seed(2018)


class Gradients:
    """
    A class for computing gradients using various methods.

    Attributes:
        n (int): The number of data points.
        x (np.ndarray): The input data.
        y (np.ndarray): The output data.
        cost_func (Callable): The type of cost function to use. Default is fast_OLS.
        analytic_derivative (Callable): Analytic derivative of cost function, if applicable.
        scheduler (Scheduler): The scheduler to use for updating the learning rate. Default is Constant.
        dim (int): The degree of the polynomial to use for the design matrix. Default is 2.
        lmbda (float): The regularization parameter. Default is None.
        polynomial (bool): Whether to use a polynomial design matrix. Default is True.
    """

    def __init__(
        self,
        n: int,
        x: np.ndarray,
        y: np.ndarray,
        cost_func: Callable = fast_OLS,
        analytic_derivative: Callable = None,
        scheduler: Scheduler = Constant,
        dim: int = 2,
        lmbda: float = None,
        polynomial: bool = True,
    ) -> None:
        if not isinstance(n, int):
            raise TypeError(f"n should be an integer, not {n} of type {type(n)}")
        if not (n == len(x) == len(y)):
            raise ValueError("Number of points must correspond to length of arrays")

        self.n = n
        self.x = x
        self.y = y
        if polynomial:
            self.X = design(x, dim, len(x))
        else:
            self.X = x
        self.lmbda = lmbda
        self.dim = dim
        self.cost_func = cost_func
        if analytic_derivative is None:
            self.gradient = jit(grad(self.cost_func, argnums=2))
        else:
            self.gradient = analytic_derivative

        self.scheduler = scheduler

    @profile
    def GradientDescent(self, theta: np.ndarray, n_iter: int) -> np.ndarray:
        """
        Performs gradient descent to optimize the model parameters.

        Args:
            theta (np.ndarray): The initial model parameters.
            n_iter (int): The number of iterations to perform.

        Returns:
            np.ndarray: The optimized model parameters.
        """
        self.errors = np.zeros(n_iter)

        for i in range(n_iter):
            gradients = self.gradient(self.X, self.y, theta, self.lmbda)
            change = self.scheduler.update_change(gradients)
            theta = update_theta(theta, change)
            tmp = self.cost_func(self.X, self.y, theta, self.lmbda)
            self.errors = assign(self.errors, i, tmp)

        return theta

    def StochasticGradientDescent(
        self, theta: np.ndarray, n_epochs: int, M: int
    ) -> np.ndarray:
        """
        Performs stochastic gradient descent to optimize the model parameters.

        Args:
            theta (np.ndarray): The initial model parameters.
            n_epochs (int): The number of epochs to perform.
            M (int): The batch size.

        Returns:
            np.ndarray: The optimized model parameters.
        """
        m = self.n // M

        self.errors = np.zeros(n_epochs)

        for epoch in range(n_epochs):
            self.scheduler.reset()
            for i in range(m):
                idxs = onp.random.choice(self.n, M)
                xi = self.X[idxs]
                yi = self.y[idxs]

                gradients = self.gradient(xi, yi, theta, self.lmbda)
                change = self.scheduler.update_change(gradients)

                theta = update_theta(theta, change)

            self.errors = assign(
                self.errors, epoch, self.cost_func(self.X, self.y, theta, self.lmbda)
            )

        return theta

    def predict(self, x: np.ndarray, theta: np.ndarray, dim: int = 2) -> np.ndarray:
        """
        Predicts the output values for the given input data using the model parameters.

        Args:
            x (np.ndarray): The input data.
            theta (np.ndarray): The model parameters.
            dim (int): The degree of the polynomial to use for the design matrix. Default is 2.

        Returns:
            np.ndarray: The predicted output values.
        """
        X = design(x, dim, len(x))
        return lax.dot(X, theta)
