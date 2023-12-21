import jax.numpy as np
from jax import lax, jit
from line_profiler import profile


class Scheduler:
    """
    Base class for Schedulers
    """

    def __init__(self, eta: float) -> None:
        """
        Initializes the scheduler with a given learning rate.

        Args:
            eta (float): The learning rate for the scheduler.
        """
        raise NotImplementedError

    def update_change(self, gradient: np.ndarray) -> None:
        """
        Updates the scheduler based on the gradient.

        Args:
            gradient (np.ndarray): The gradient used to update the scheduler.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Resets the scheduler to its initial state.
        """
        pass


class Constant(Scheduler):
    """
    A learning rate scheduler that keeps the learning rate constant throughout training.

    Args:
        eta (float): The learning rate.

    Attributes:
        eta (float): The learning rate.

    Methods:
        update_change: Updates the learning rate by multiplying it with the gradient.

    """

    def __init__(self, eta: float) -> None:
        self.eta = eta

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Updates the learning rate by multiplying it with the gradient.

        Args:
            gradient (np.ndarray): The gradient.

        Returns:
            np.ndarray: The updated learning rate.

        """
        return fast_const(self.eta, gradient)


class Momentum(Scheduler):
    """
    Implements the Momentum optimizer.

    Args:
        eta (float): The learning rate.
        momentum (float): The momentum parameter.

    Attributes:
        eta (float): The learning rate.
        momentum (float): The momentum parameter.
        change (float): The change in the weights.

    Methods:
        update_change: Updates the change in the weights.

    """

    def __init__(self, eta: float, rho: float) -> None:
        self.eta = eta
        self.momentum = rho
        self.change = 0.0

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Updates the change in the weights.

        Args:
            gradient (np.ndarray): The gradient of the weights.

        Returns:
            np.ndarray: The updated change in the weights.

        """
        self.change = fast_mom(self.momentum, self.change, self.eta, gradient)
        return self.change


class Adagrad(Scheduler):
    """
    Adagrad optimizer.

    Args:
        eta (float): Learning rate.

    Attributes:
        eta (float): Learning rate.
        G_t (ndarray): Matrix of sum of squares of past gradients.

    Methods:
        update_change(gradient: np.ndarray) -> np.ndarray:
            Update the weights of the model based on the gradient.
        reset() -> None:
            Reset the optimizer's state.
    """

    def __init__(self, eta: float) -> None:
        self.eta = eta
        self.G_t = None

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update the weights of the model based on the gradient.

        Args:
            gradient (ndarray): Gradient of the loss function.

        Returns:
            ndarray: Updated weights of the model.
        """
        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t = fast_gt_add(gradient, self.G_t)
        G_t_inverse = fast_gt_inv(self.G_t)

        change = fast_adagrad(self.eta, gradient, G_t_inverse)
        return change

    def reset(self) -> None:
        """
        Reset the optimizer's state.
        """
        self.G_t = None


class AdagradMomentum(Scheduler):
    """
    AdagradMomentum is a class that implements the Adagrad Momentum optimizer.

    Args:
        eta (float): The learning rate.
        momentum (float): The momentum parameter.

    Attributes:
        eta (float): The learning rate.
        G_t (np.ndarray): The sum of the squares of the gradients.
        momentum (float): The momentum parameter.
        change (np.ndarray): The change in the weights.

    Methods:
        update_change(gradient: np.ndarray) -> np.ndarray:
            Updates the change in the weights based on the gradient.
        reset() -> None:
            Resets the sum of the squares of the gradients to None.
    """

    def __init__(self, eta: float, rho: float) -> None:
        self.eta = eta
        self.G_t = None
        self.momentum = rho
        self.change = 0.0

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Updates the change in the weights based on the gradient.

        Args:
            gradient (np.ndarray): The gradient.

        Returns:
            np.ndarray: The change in the weights.
        """
        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t = fast_gt_add(gradient, self.G_t)

        G_t_inverse = fast_gt_inv(self.G_t)

        self.change = fast_adamoment(
            self.eta, gradient, G_t_inverse, self.momentum, self.change
        )
        return self.change

    def reset(self) -> None:
        """
        Resets the sum of the squares of the gradients to None.
        """
        self.G_t = None


class RMS_prop(Scheduler):
    """
    Root Mean Square Propagation (RMS_prop) optimizer.

    Args:
        eta (float): Learning rate.
        rho (float): Decay rate.

    Attributes:
        eta (float): Learning rate.
        rho (float): Decay rate.
        second (float): Running average of the square of the gradients.

    Methods:
        update_change(gradient: np.ndarray) -> np.ndarray:
            Update the parameters based on the gradient.
        reset() -> None:
            Reset the running average of the square of the gradients.
    """

    def __init__(self, eta: float, rho: float) -> None:
        self.eta = eta
        self.rho = rho
        self.second = 0.0

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update the parameters based on the gradient.

        Args:
            gradient (np.ndarray): Gradient of the loss function.

        Returns:
            np.ndarray: Updated parameters.
        """
        self.second = fast_rms_second(self.rho, self.second, gradient)
        change = fast_rmsprop(self.eta, gradient, self.second)
        return change

    def reset(self) -> None:
        """
        Reset the running average of the square of the gradients.
        """
        self.second = 0.0


class Adam(Scheduler):
    """
    Adam optimizer.

    Args:
        eta (float): Learning rate.
        rho (float): Decay rate for the first moment estimate.
        rho2 (float): Decay rate for the second moment estimate.

    Attributes:
        moment (float): First moment estimate.
        second (float): Second moment estimate.
        n_epochs (int): Number of epochs.

    Methods:
        update_change: Update the parameters.
        reset: Reset the optimizer.

    """

    def __init__(self, eta: float, rho: float, rho2: float) -> None:
        self.eta = eta
        self.rho = rho
        self.rho2 = rho2
        self.moment: float = 0.0
        self.second: float = 0.0
        self.n_epochs: int = 1

    @profile
    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update the parameters.

        Args:
            gradient (np.ndarray): Gradient of the loss function.

        Returns:
            np.ndarray: Updated parameters.
        """
        self.moment = fast_adam_moment(self.rho, self.moment, gradient)
        self.second = fast_adam_second(self.rho2, self.second, gradient)

        change = fast_adam(
            self.moment,
            self.rho,
            self.second,
            self.rho2,
            self.eta,
            self.n_epochs,
        )
        return change

    def reset(self) -> None:
        """
        Reset the optimizer.
        """
        self.n_epochs += 1
        self.moment = 0.0
        self.second = 0.0


class TimeDecay(Scheduler):
    """
    A scheduler that applies time decay to the learning rate.

    Attributes:
        t0 (float): The initial time value.
        t1 (float): The final time value.
        minibatch_size (float): The size of the minibatch.
        epochs (float): The number of epochs.
        minibatch_num (float): The number of minibatches.
    """

    def __init__(self, t0: float, t1: float, minibatch_size: int):
        # Converts to float for jax.jit
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.minibatch_size = float(minibatch_size)
        self.epochs = 0.0
        self.minibatch_num = 0.0

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Updates the learning rate based on the current epoch and minibatch number.

        Args:
            gradient (np.ndarray): The gradient.

        Returns:
            np.ndarray: The updated gradient.
        """
        change = fast_timedecay(
            self.epochs,
            self.minibatch_size,
            self.minibatch_num,
            self.t0,
            self.t1,
            gradient,
        )
        self.minibatch_num += 1
        return change

    def reset(self) -> None:
        """
        Resets the scheduler for the next epoch.
        """
        self.epochs += 1
        self.minibatch_num = 0.0


@jit
def fast_const(eta: float, gradient: np.ndarray) -> np.ndarray:
    """
    Computes the product of the learning rate and the gradient.

    This function, along with the others like it, is used to speed up the computation. Due
    to the way JAX's jit works, methods need to be completely functional in order to compile
    properly. This means that we cannot use class attributes in the methods, and instead need
    to pass them as arguments. Due to the complexity of the different methods, and the need
    to keep track of previous values, we opted to extract the computationally heavy parts for
    compilation, and leave the bookkeeping to the methods. LAX is not particularly readable,
    but it is much faster than the alternatives.

    Args:
        eta (float): The learning rate.
        gradient (np.ndarray): The gradient of the loss function.

    Returns:
        np.ndarray: The product of the learning rate and the gradient.
    """
    return lax.mul(eta, gradient)


@jit
def fast_mom(
    momentum: np.ndarray, change: float, eta: float, gradient: np.ndarray
) -> np.ndarray:
    """
    Computes the momentum update for a given gradient.

    Args:
        momentum (np.ndarray): Array representing the momentum.
        change (float): A float representing the change in momentum.
        eta (float): A float representing the learning rate.
        gradient (np.ndarray): Array representing the gradient.

    Returns:
        numpy.ndarray: A numpy array containing the updated momentum.
    """
    return lax.add(
        lax.mul(momentum, change),
        lax.mul(eta, gradient),
    )


@jit
def fast_gt_add(gradient: np.ndarray, G_t: np.ndarray) -> np.ndarray:
    """
    Computes the sum of the outer product of the gradient with itself and the input matrix G_t.

    Args:
        gradient (np.ndarray): A 1D array of shape (n,) representing the gradient.
        G_t (np.ndarray): A 2D array of shape (n, n) representing the matrix to be updated.

    Returns:
        np.ndarray: A 2D array of shape (n, n) representing the updated matrix.
    """
    G_t = lax.add(
        G_t,
        lax.dot(gradient, gradient.T),
    )
    return G_t


@jit
def fast_adagrad(
    eta: float, gradient: np.ndarray, G_t_inverse: np.ndarray
) -> np.ndarray:
    """
    Computes the change in the parameters using the fast Adagrad algorithm.

    Args:
        eta (float): The learning rate.
        gradient (np.ndarray): The gradient of the loss function with respect to the parameters.
        G_t_inverse (np.ndarray): The inverse of the sum of the squares of the gradients up to the current time step.

    Returns:
        np.ndarray: The change in the parameters.
    """
    change = lax.mul(
        lax.mul(
            eta,
            gradient,
        ),
        G_t_inverse,
    )
    return change


@jit
def fast_adamoment(
    eta: float,
    gradient: np.ndarray,
    G_t_inverse: np.ndarray,
    momentum: float,
    change: np.ndarray,
) -> np.ndarray:
    """
    Computes the update for the change in the parameters using the AdagradMomentum algorithm,
    extracted for compatibility with jax.jit.

    Args:
        eta (float):
            The learning rate.
        gradient (np.ndarray):
            The gradient of the loss function with respect to the parameters.
        G_t_inverse (np.ndarray):
            The inverse of the diagonal matrix containing the running average of the squared gradients.
        momentum (float):
            The momentum parameter.
        change (np.ndarray):
            The current change in the parameters.

    Returns:
        np.ndarray: The updated change in the parameters.
    """
    change = lax.add(
        lax.mul(
            momentum,
            change,
        ),
        lax.mul(
            lax.mul(eta, gradient),
            G_t_inverse,
        ),
    )
    return change


@jit
def fast_gt_inv(G_t: np.ndarray) -> np.ndarray:
    """
    Computes the inverse of a matrix G_t using JAX's jit and lax libraries.

    Args:
        G_t (np.ndarray): Array of shape (n, n) representing the matrix to be inverted.

    Returns:
        np.ndarray: Array of shape (n, n) representing the inverse of G_t.
    """
    delta = 1e-7  # avoid division by zero
    G_t_inverse = lax.reciprocal(
        delta
        + lax.sqrt(
            np.reshape(
                np.diagonal(G_t),
                (G_t.shape[0], 1),
            ),
        )
    )
    return G_t_inverse


@jit
def fast_rms_second(rho: float, second: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """
    Computes the exponentially weighted moving average of the squared gradient.

    Args:
        rho (float): A float representing the decay rate of the moving average.
        second (np.ndarray): Array representing the previous moving average.
        gradient (np.ndarray): Array representing the gradient.

    Returns:
        np.ndarray: Array representing the updated moving average.
    """
    second = lax.add(
        lax.mul(rho, second),
        lax.mul(
            lax.sub(1.0, rho),
            lax.square(gradient),
        ),
    )
    return second


@jit
def fast_rmsprop(eta: float, gradient: np.ndarray, second: np.ndarray) -> np.ndarray:
    """
    Computes the RMSProp update for a given gradient and second moment estimate.

    Args:
        eta (float): A float representing the learning rate.
        gradient (np.ndarray): Array representing the gradient.
        second (np.ndarray): Array representing the second moment estimate.

    Returns:
        np.ndarray: Array representing the RMSProp update.
    """
    delta = 1e-7
    change = lax.div(
        lax.mul(eta, gradient),
        lax.sqrt(
            lax.add(second, delta),
        ),
    )
    return change


@jit
def fast_adam_moment(
    rho: float, moment: np.ndarray, gradient: np.ndarray
) -> np.ndarray:
    """
    Computes the exponentially weighted moving average of the gradient using the Adam optimizer.

    Args:
        rho (float): The exponential decay rate.
        moment (np.ndarray): The exponentially weighted moving average of the gradient.
        gradient (np.ndarray): The gradient.

    Returns:
        moment (np.ndarray): The updated exponentially weighted moving average of the gradient.
    """
    moment = lax.add(
        lax.mul(rho, moment),
        lax.mul(
            lax.sub(1.0, rho),
            gradient,
        ),
    )
    return moment


@jit
def fast_adam_second(
    rho2: float, second: np.ndarray, gradient: np.ndarray
) -> np.ndarray:
    """
    Computes the second moment estimate in the Adam optimizer.

    Args:
        rho2 (float): A float representing the exponential decay rate for the second moment estimate.
        second (np.ndarray): Array representing the second moment estimate.
        gradient (np.ndarray): Array representing the gradient.

    Returns:
        np.ndarray: Array representing the updated second moment estimate.
    """
    second = lax.add(
        lax.mul(rho2, second),
        lax.mul(
            lax.sub(1.0, rho2),
            lax.square(gradient),
        ),
    )
    return second


@jit
def fast_adam(
    moment: np.ndarray,
    rho: float,
    second: np.ndarray,
    rho2: float,
    eta: float,
    n_epochs: int,
) -> np.ndarray:
    """
    Computes the change in the parameters using the fast Adam optimization algorithm.

    Args:
        moment (np.ndarray): Array of shape (batch_size, num_params) of the first moment estimate.
        rho (float): A float representing the exponential decay rate for the first moment estimate.
        second (np.ndarray): Array of shape (batch_size, num_params) representing the second moment estimate.
        rho2 (float): A float representing the exponential decay rate for the second moment estimate.
        eta (float): A float representing the learning rate.
        n_epochs (int): An integer representing the number of epochs.

    Returns:
        np.ndarray: A numpy array of shape (batch_size, num_params) representing the change in the parameters.
    """
    delta = 1e-7  # avoid division by zero
    moment_corrected = lax.div(
        moment,
        lax.sub(
            lax.add(1.0, delta),
            # lax.integer_pow(rho, n_epochs),
            lax.pow(rho, n_epochs),
        ),
    )
    second_corrected = lax.div(
        second,
        lax.sub(
            lax.add(1.0, delta),
            # lax.integer_pow(rho2, n_epochs),
            lax.pow(rho2, n_epochs),
        ),
    )
    change = lax.div(
        lax.mul(eta, moment_corrected),
        lax.sqrt(
            lax.add(second_corrected, delta),
        ),
    )
    return change


@jit
def fast_timedecay(
    epochs: float,
    minibatch_size: float,
    minibatch_num: float,
    t0: float,
    t1: float,
    gradient: np.ndarray,
) -> np.ndarray:
    """
    Computes the learning rate for a given epoch and minibatch using the
    time decay schedule. Seperated into a function to be used with jax.jit.

    Args:
        epochs (float): The total number of epochs.
        minibatch_size (float): The size of the minibatch.
        minibatch_num (float): The current minibatch number.
        t0 (float): The initial learning rate.
        t1 (float): The decay rate.
        gradient (np.ndarray): The gradient of the loss function.

    Returns:
        np.ndarray: The change in the learning rate for the current epoch and minibatch.
    """
    t = lax.add(lax.mul(epochs, minibatch_size), minibatch_num)
    eta = lax.div(t0, lax.add(t, t1))
    change = lax.mul(eta, gradient)
    return change
