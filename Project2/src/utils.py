import jax.numpy as np
from jax import jit, lax
from functools import partial


@jit
def assign(arr, idx, val):
    arr = arr.at[idx].set(val)
    return arr


@jit
def assign_row(arr, idx, val):
    arr = arr.at[idx, :].set(val)
    return arr


@jit
def vstack_arrs(arr1, arr2):
    return np.vstack([arr1, arr2])


@partial(jit, static_argnums=(1, 2))
def design(x: np.ndarray, dim: int, n: int) -> np.ndarray:
    """
    Computes the design matrix for the given input data.

    Args:
        x (np.ndarray): The input data.
        dim (int): The degree of the polynomial to use for the design matrix.
        n (int): The number of data points. Default is None.

    Returns:
        np.ndarray: The design matrix.
    """
    X = np.ones((n, dim + 1))
    for i in range(1, dim + 1):
        X = X.at[:, i].set((x**i).ravel())

    return X


@jit
def update_theta(theta: np.ndarray, change: np.ndarray):
    # return theta - change
    return lax.sub(theta, change)
