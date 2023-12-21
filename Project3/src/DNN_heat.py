import jax.numpy as np
from jax import jacobian, hessian, grad, jit, vmap, lax
import numpy as onp
from jax.tree_util import Partial
from typing import Callable
from tqdm import tqdm
from pathlib import Path
from plotutils import plot_surface, plot_at_timestep


@jit
def deep_neural_network(
    deep_params: list[np.ndarray],
    x: np.ndarray,
    activation_func: Callable[[float], float],
) -> float:
    # x is now a point and a 1D numpy array; make it a column vector
    num_coordinates = np.size(x, 0)
    x = x.reshape(num_coordinates, -1)

    num_points = np.size(x, 1)
    # N_hidden is the number of hidden layers
    # -1 since params consist of parameters to all the hidden layers AND the output layer
    N_hidden = len(deep_params) - 1

    # Assume that the input layer does nothing to the input x
    x_input = x
    x_prev = x_input

    # Iterate through the hidden layers:
    for layer in range(N_hidden):
        # From the list of parameters P; find the correct weigths and bias for this layer
        w_hidden = deep_params[layer]

        # Add a row of ones to include bias
        x_prev = np.concatenate((np.ones((1, num_points)), x_prev), axis=0)

        z_hidden = lax.dot(w_hidden, x_prev)
        x_hidden = activation_func(z_hidden)

        # Update x_prev such that next layer can use the output from this layer
        x_prev = x_hidden

    # Output layer:

    # Get the weights and bias for this layer
    w_output = deep_params[-1]

    # Include bias:
    x_prev = np.concatenate((np.ones((1, num_points)), x_prev), axis=0)

    z_output = lax.dot(w_output, x_prev)
    x_output = z_output

    # Unpack the final layer
    return x_output[0][0]


# Define the trial solution and cost function
@jit
def u(x: float) -> float:
    # sin(pi * x)
    return lax.sin(lax.mul(np.pi, x))


@jit
def g_trial(
    point: np.ndarray, P: list[np.ndarray], activation_func: Callable[[float], float]
) -> float:
    x, t = point
    # (1 - t) * u(x) + x * (1 - x) * t * N(x, t, P)
    res = lax.mul(
        lax.sub(1.0, t),
        u(x),
    ) + lax.mul(
        lax.mul(x, lax.sub(1.0, x)),
        lax.mul(t, deep_neural_network(P, point, activation_func)),
    )
    return res


@jit
def innercost(
    point: np.ndarray, P: list[np.ndarray], activation_func: Callable[[float], float]
) -> float:
    # The inner cost function, evaluating each point
    g_t_jacobian_func = jit(jacobian(g_trial))
    g_t_hessian_func = jit(hessian(g_trial))

    g_t_jacobian = g_t_jacobian_func(point, P, activation_func)
    g_t_hessian = g_t_hessian_func(point, P, activation_func)

    # partial w.r.t. t
    g_t_dt = g_t_jacobian[1]
    # partial^2 w.r.t. x
    g_t_d2x = g_t_hessian[0][0]

    # (g_t_dt - g_t_d2x)^2
    err_sqr = lax.integer_pow(lax.sub(g_t_dt, g_t_d2x), 2)

    return err_sqr


# The cost function:
@jit
def cost_function(
    P: list[np.ndarray],
    x: np.ndarray,
    t: np.ndarray,
    activation_func: Callable[[float], float],
) -> float:
    X, T = np.meshgrid(x, t)
    total_points = np.vstack([X.ravel(), T.ravel()]).T
    vec_cost = vmap(innercost, (0, None, None), 0)
    cost_sum = np.sum(vec_cost(total_points, P, activation_func))
    return cost_sum / (np.size(x) * np.size(t))


# For comparison, define the analytical solution
@jit
def g_analytic(point: np.ndarray) -> float:
    x, t = point
    # e^(-pi^2 * t) * sin(pi * x)
    res = lax.mul(
        lax.exp(
            lax.mul(-lax.integer_pow(np.pi, 2), t),
        ),
        lax.sin(
            lax.mul(np.pi, x),
        ),
    )
    return res


@jit
def update_P(P: np.ndarray, cost_grad: np.ndarray, lmb: float) -> np.ndarray:
    return lax.sub(P, lax.mul(lmb, cost_grad))


# Set up a function for training the network to solve for the equation
def solve_pde_deep_neural_network(
    x: np.ndarray,
    t: np.ndarray,
    num_neurons: list[int],
    num_iter: int,
    lmb: float,
    activation_func: Callable[[float], float],
) -> list[np.ndarray]:
    # Set up initial weigths and biases
    N_hidden = np.size(num_neurons)

    # Set up initial weigths and biases

    # Initialize the list of parameters:
    P: list[np.ndarray] = [None] * (N_hidden + 1)  # + 1 to include the output layer

    # 2 since we have two points, +1 to include bias
    P[0] = onp.random.randn(num_neurons[0], 2 + 1)

    for layer in range(1, N_hidden):
        # +1 to include bias
        P[layer] = onp.random.randn(num_neurons[layer], num_neurons[layer - 1] + 1)

    # For the output layer
    # +1 since bias is included
    P[-1] = onp.random.randn(1, num_neurons[-1] + 1)

    print("Initial cost: ", cost_function(P, x, t, activation_func))

    cost_function_grad = jit(grad(cost_function, 0))

    for _ in tqdm(range(num_iter)):
        cost_grad = cost_function_grad(P, x, t, activation_func)

        for layer in range(N_hidden + 1):
            P[layer] = update_P(P[layer], cost_grad[layer], lmb)

    print(f"Final cost: {cost_function(P, x, t, activation_func)}")

    return P


def main(
    activation_func: Callable[[float], float],
    num_hidden_neurons: list[int],
    lmb: float = 0.01,
    save: bool = False,
    savePath: Path = None,
) -> None:
    # Use the neural network:
    func_name = activation_func.__name__
    saveBase = f"{round(np.log10(lmb))}_{num_hidden_neurons}"
    onp.random.seed(15)
    if savePath is None:
        savePath = Path(__file__).parent.parent / f"figures/{func_name}"

    savePath.mkdir(parents=True, exist_ok=True)

    # Decide the vales of arguments to the function to solve
    Nx = 30
    Nt = 30
    x = np.linspace(0, 1, Nx)
    t = np.linspace(0, 1, Nt)

    # Set up the parameters for the network
    num_iter = 20000
    # lmb = 0.01

    par_func = Partial(activation_func)

    P = solve_pde_deep_neural_network(x, t, num_hidden_neurons, num_iter, lmb, par_func)

    # Store the results
    Nx = 201
    Nt = 201
    x = np.linspace(0, 1, Nx)
    t = np.linspace(0, 1, Nt)

    T, X = np.meshgrid(t, x)
    total_points = np.vstack([X.ravel(), T.ravel()]).T

    g_dnn_ag = vmap(g_trial, (0, None, None), 0)(total_points, P, par_func)
    g_dnn_ag = g_dnn_ag.reshape(Nt, Nx)
    G_analytical = vmap(g_analytic, 0)(total_points)
    G_analytical = G_analytical.reshape(Nt, Nx)

    # Find the map difference between the analytical and the computed solution
    diff_ag = np.abs(g_dnn_ag - G_analytical)
    print(
        f"Max absolute difference between the analytical solution and the network: {np.max(diff_ag):g}"
    )

    # Plot the solutions in two dimensions, that being in position and time

    title = f"{func_name}: Solution from the deep neural network w/ {len(num_hidden_neurons)} layers"
    plot_surface(T, X, g_dnn_ag, title, save, savePath, f"{saveBase}_dnn")

    title = f"{func_name}: Analytical solution"
    plot_surface(T, X, G_analytical, title, save, savePath, "Analytical")

    title = f"{func_name}: Difference"
    plot_surface(T, X, diff_ag, title, save, savePath, f"{saveBase}_diff")

    # Take some slices of the 3D plots just to see the solutions at particular times
    indicies = [0, int(Nt / 2), Nt - 1]
    for index in indicies:
        timestep = t[index]
        res_dnn = g_dnn_ag[:, index]
        res_analytic = G_analytical[:, index]
        plot_at_timestep(
            x,
            res_dnn,
            res_analytic,
            timestep,
            func_name,
            save,
            savePath,
            saveBase,
        )

    # plt.show()


if __name__ == "__main__":
    from jax.nn import relu, sigmoid, tanh, leaky_relu, swish, elu

    relu.__name__ = "ReLU"
    sigmoid.__name__ = "Sigmoid"
    tanh.__name__ = "Tanh"
    leaky_relu.__name__ = "Leaky_ReLU"
    swish.__name__ = "Swish"
    elu.__name__ = "ELU"

    functions = [relu, sigmoid, tanh, leaky_relu, swish, elu]
    lmbs = [0.0001, 0.01, 0.01, 0.0001, 0.001, 0.0001]
    # for func, lmb in zip(functions, lmbs):
    #     main(func, [100, 25], lmb=lmb, save=True)

    # main(swish, [50, 50, 50, 50], lmb=0.0000001, save=True)
