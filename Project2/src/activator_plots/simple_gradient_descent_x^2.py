import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plotutils import setup_axis, setColors

path = Path(__file__).parent.parent.parent / "figures"


# Define the function
def f(x):
    return x**2


# Define the derivative of the function
def df(x):
    return 2 * x


# Perform gradient descent
def gradient_descent(starting_point, learning_rate, num_iterations):
    points = np.zeros(num_iterations)
    points[0] = starting_point

    for i in range(num_iterations - 1):
        current_point = points[i]
        gradient = df(current_point)
        new_point = current_point - learning_rate * gradient
        points[i + 1] = new_point

    return points


# Set the hyperparameters
learning_rate = 0.1
num_iterations = 30
starting_point = -4

# Perform gradient descent and store the points
points = gradient_descent(starting_point, learning_rate, num_iterations)

# Plot the function and the gradient descent path
x = np.linspace(-5, 5, 100)
y = f(x)

ax = setup_axis(xlim=[-5.1, 5.1], ylim=[-1, 25.2])
ax.set_aspect("auto")

iteration_counter = np.arange(1, num_iterations + 1)
cmap, norm, sm = setColors(iteration_counter, cmap_name="viridis", norm_type="linear")

ax.scatter(
    points,
    f(points),
    color=cmap(norm(iteration_counter)),
    label="Gradient Descent Path",
    zorder=2.5,
    s=100,
)
ax.plot(x, y, color="C0", label="$f(x) = x^2$", linewidth=3)
plt.legend(loc="upper left", bbox_to_anchor=(0.5, 0.93), fontsize=12)
plt.title(r"Gradient Descent on $f(x) = x^2$", fontsize=15)
plt.xlabel(r"$x$", fontsize=12)
plt.ylabel(r"$y$", fontsize=12, rotation=0)

cbar = plt.colorbar(sm, ax=ax)
cbar.ax.set_ylabel("Iteration number", fontsize=12)
index = np.searchsorted(points, -1e-2, side="right")
print(index)

plt.savefig(path / "simple_gradient_descent_x^2.pdf", bbox_inches="tight")
