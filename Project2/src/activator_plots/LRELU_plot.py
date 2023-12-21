import numpy as np
from plotutils import setup_axis
from Activators import ReLU
import matplotlib.pyplot as plt
from pathlib import Path

path = Path(__file__).parent.parent.parent / "figures/activators"

ax = setup_axis(xlim=[-2, 1], ylim=[-0.5, 1])

x = np.linspace(-2, 2, 100)


def exaggerated_LRELU(x):
    return np.where(x > 0, x, 0.1 * x)


ax.plot(x, ReLU(x), label="ReLU", linewidth=3)
ax.plot(x, exaggerated_LRELU(x), "--", label="LReLU")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$", rotation=0)
ax.set_title(r"Leaky ReLU ($\gamma = 0.1$) against ReLU")
ax.legend()

plt.savefig(path / "LReLU.pdf", bbox_inches="tight")
