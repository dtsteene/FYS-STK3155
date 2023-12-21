from jax import grad, vmap, lax, jit
import jax.numpy as np
from typing import Callable


@jit
def identity(X: np.ndarray) -> np.ndarray:
    """
    Identity activation function.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: The input array, unchanged.
    """
    return X


@jit
def sigmoid(X: np.ndarray) -> np.ndarray:
    """
    Applies the sigmoid activation function to the input array.

    Args:
        X (np.ndarray): The input array.

    Returns:
        np.ndarray: The output array after applying the sigmoid function.
    """
    return lax.reciprocal(lax.add(1.0, lax.exp(-X)))


@jit
def zero_one_clip(X: np.ndarray) -> np.ndarray:
    """
    Applies the zero-one clip activation function to the input array.

    Args:
        X (np.ndarray): The input array.

    Returns:
        np.ndarray: The output array after applying the zero-one clip function.
    """
    return np.clip(X, 0, 1)


@jit
def softmax(X: np.ndarray) -> np.ndarray:
    """
    Computes the softmax activation function for a given input array.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying the softmax function.
    """
    X = X - np.max(X, axis=-1, keepdims=True)
    delta = 10e-10
    return np.exp(X) / (np.sum(np.exp(X), axis=-1, keepdims=True) + delta)


@jit
def ReLU(X: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation function.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array with values equal to X where X > 0, and 0 elsewhere.
    """
    return np.where(
        X > np.zeros(X.shape, dtype=float), X, np.zeros(X.shape, dtype=float)
    )


@jit
def LRELU(X: np.ndarray) -> np.ndarray:
    """
    Leaky Rectified Linear Unit activation function.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array with the same shape as X.
    """
    delta = 10e-4
    return np.where(X > np.zeros(X.shape), X, delta * X)


def derivate(func: Callable) -> Callable:
    """
    Computes the derivative of the input activation function.

    Args:
        func: The activation function to compute the derivative of.

    Returns:
        The derivative of the input activation function.
    """
    if func.__name__ == "ReLU":

        @jit
        def func(X: np.ndarray) -> np.ndarray:
            """
            Computes the derivative of the ReLU activation function.

            Args:
                X: The input to the ReLU activation function.

            Returns:
                The derivative of the ReLU activation function.
            """
            return np.where(X > 0, 1.0, 0.0)

        return func

    elif func.__name__ == "LRELU":

        @jit
        def func(X: np.ndarray) -> np.ndarray:
            """
            Computes the derivative of the Leaky ReLU activation function.

            Args:
                X: The input to the Leaky ReLU activation function.

            Returns:
                The derivative of the Leaky ReLU activation function.
            """
            delta = 10e-4
            return np.where(X > 0, 1.0, delta)

        return func

    else:
        return jit(vmap(vmap(grad(func))))
