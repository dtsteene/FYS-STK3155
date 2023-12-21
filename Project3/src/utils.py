import jax.numpy as jnp
from jax import jit, lax, grad
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
    return jnp.vstack([arr1, arr2])


@partial(jit, static_argnums=(1, 2))
def design(x: jnp.ndarray, dim: int, n: int) -> jnp.ndarray:
    """
    Computes the design matrix for the given input data.

    Args:
        x (np.ndarray): The input data.
        dim (int): The degree of the polynomial to use for the design matrix.
        n (int): The number of data points. Default is None.

    Returns:
        np.ndarray: The design matrix.
    """
    X = jnp.ones((n, dim + 1))
    for i in range(1, dim + 1):
        X = X.at[:, i].set((x**i).ravel())

    return X


@jit
def update_theta(theta: jnp.ndarray, change: jnp.ndarray):
    # return theta - change
    return lax.sub(theta, change)


def NNu_unpacked(NNu):
    def unpacked(t, x, theta):
        # .item() extracts the scalar output
        return NNu(jnp.stack([t, x], axis=-1), theta)[0, 0]
    return unpacked


def unpack(f):
    def unpacked(t, x, theta):
        return f(t, x, theta)[0]
    return unpacked


def make_d2_dx2_d_dt(NNu):
    dNNu_dx = jit(unpack(grad(NNu, 1)))
    d2NNu_dx2 = jit(unpack(grad(dNNu_dx, 1)))
    dNNu_dt = jit(unpack(grad(NNu, 0)))

    return d2NNu_dx2, dNNu_dt
