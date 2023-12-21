from Data import TerrainData
from pathlib import Path
from Models import Lasso
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


terrainfile = Path(__file__).parent.parent / "DataFiles" / "SRTM_data_Norway_1.tif"


data = TerrainData(
    terrainfile,
    40,
    15,
    savePlots=True,
    showPlots=True,
    figsPath=Path(".").parent,
)
model = Lasso(lmbd=10 ** (-4))

X = model.create_X(data.x_, data.y_, 13)
model.fit(X, data.z_, lmbd=0.00001)

N = 40
eval_x = np.linspace(0, 1, N)
eval_y = np.linspace(0, 1, N)

eval_x, eval_y = np.meshgrid(eval_x, eval_y)
flat_x = eval_x.flatten().reshape(-1, 1)
flat_y = eval_y.flatten().reshape(-1, 1)

pred_X = model.create_X(flat_x, flat_y, 13)
pred_z = model.predict(pred_X).reshape(N, -1)


fig = plt.figure()  # figsize=plt.figaspect(0.5))
ax = fig.add_subplot(projection="3d")

surf = ax.plot_surface(
    eval_x,
    eval_y,
    pred_z,
    linewidth=0,
    cmap=cm.coolwarm,
    antialiased=True,
)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
fig.suptitle("Predicted terrain shape")

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.savefig(
    Path(__file__).parent.parent / "figures/Terrain/PredictedSurface.png", dpi=300
)
plt.show()
