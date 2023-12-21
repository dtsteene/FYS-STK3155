import numpy as np
from plotutils import setup_axis
from Activators import sigmoid, zero_one_clip
import matplotlib.pyplot as plt
from pathlib import Path

path = Path(__file__).parent.parent.parent / "figures/activators"

ax = setup_axis(xlim=[-4, 4], ylim=[-0.1, 1.1])

x = np.linspace(-4, 4, 100)

ax.plot(x, sigmoid(x), label="Sigmoid")
ax.plot(x, zero_one_clip(x), label="(0,1)-clip")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$", rotation=0)
ax.set_title("Sigmoid Activation Function")
plt.legend()

plt.savefig(path / "sigmoid.pdf", bbox_inches="tight")
