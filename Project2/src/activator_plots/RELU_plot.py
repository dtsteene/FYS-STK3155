import numpy as np
from plotutils import setup_axis
from Activators import ReLU
import matplotlib.pyplot as plt
from pathlib import Path

path = Path(__file__).parent.parent.parent / "figures/activators"

ax = setup_axis(xlim=[-2, 2], ylim=[-0.5, 2])

x = np.linspace(-2, 2, 100)
y = ReLU(x)

ax.plot(x, y)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$", rotation=0)
ax.set_title("RELU Activation Function")

plt.savefig(path / "RELU.pdf", bbox_inches="tight")
