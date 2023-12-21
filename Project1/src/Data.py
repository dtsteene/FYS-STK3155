import matplotlib.pyplot as plt
import numpy as np


from sklearn.model_selection import train_test_split
from matplotlib import cm
from pathlib import Path
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from imageio import imread

# from sklearn.preprocessing import StandardScaler


class Data:
    "Base class for data."

    def __init__(self):
        raise NotImplementedError


class FrankeData(Data):
    "Class for holding all data relevant to Frank's function."

    def __init__(
        self,
        numPoints: int,
        alphNoise: float,
        maxDim: int,
        savePlots: bool = False,
        showPlots: bool = True,
        figsPath: Path = None,
    ) -> None:
        """Initialize the data from Franke's function

        inputs:
            numPoints (int): Number of points to generate
            alphNoise (float): Amount of noise to add
            maxDim (int): Maximal polynomial dimension

        """
        self.N = numPoints
        self.alphNoise = alphNoise
        self.maxDim = maxDim

        self.savePlots = savePlots
        self.showPlots = showPlots
        self.figsPath = figsPath

        self.x_, self.y_, self.z_ = self.generate_data(self.N, self.alphNoise)

        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
            self.z_train,
            self.z_test,
        ) = train_test_split(self.x_, self.y_, self.z_, test_size=0.2)

    def generate_data(
        self, N: int, alph: float = 0.2
    ) -> tuple[np.array, np.array, np.array]:
        """Generate the data used for testing Franke.

        inputs:
            N (int): number of points
            alph (float): amount of random noise
        returns:
            x, y, z
        """
        x_ = np.linspace(0, 1, N)
        y_ = np.linspace(0, 1, N)

        self.x_raw, self.y_raw = np.meshgrid(x_, y_)
        x = self.x_raw.flatten().reshape(-1, 1)
        y = self.y_raw.flatten().reshape(-1, 1)

        self.z_raw = self.FrankeFunction(self.x_raw, self.y_raw)
        self.z_noise = self.z_raw + alph * np.random.randn(N, N)  # * self.z_raw.mean()

        z = self.z_noise.flatten().reshape(-1, 1)

        return x, y, z

    @staticmethod
    def FrankeFunction(x: np.array, y: np.array) -> np.array:
        """Franke's function for evaluating methods.

        inputs:
            x (np.array): values in x direction

            y (np.array): values in y direction

        returns:
            (np.array) values in z direction
        """

        term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
        term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
        term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
        term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
        return term1 + term2 + term3 + term4

    def plotSurface(self) -> None:
        "Plot the surface of the data, with and without noise."
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection="3d")
        surf = ax.plot_surface(
            self.x_raw,
            self.y_raw,
            self.z_raw,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )

        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        ax.set_title("No noise")

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=10)

        ax = fig.add_subplot(1, 2, 2, projection="3d")
        surf = ax.plot_surface(
            self.x_raw,
            self.y_raw,
            self.z_noise,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )

        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        ax.set_title("With noise")

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=10)

        fig.suptitle("Franke's function")

        plt.tight_layout()

        if self.savePlots:
            plt.savefig(self.figsPath / "FrankesFunction.png", dpi=300)
        if self.showPlots:
            plt.show()
        plt.close()


class TerrainData(Data):
    def __init__(
        self,
        terrainfile: Path,
        numPoints: int,
        maxDim: int,
        savePlots: bool = False,
        showPlots: bool = True,
        figsPath: Path = None,
    ) -> None:
        """Initialize the data from terrain.

        inputs:
            terrainData (numpy.Ndarray): Original dataset
            numPoints (int): Number of points to generate.
            maxDim (int): Maximal polynomial dimension
        """
        self.terrainData = np.asarray(imread(terrainfile))
        self.N = numPoints
        self.maxDim = maxDim
        self.savePlots = savePlots
        self.showPlots = showPlots
        self.figsPath = figsPath

        self.x_, self.y_, self.z_ = self.ready_data(self.N)

        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
            self.z_train,
            self.z_test,
        ) = train_test_split(self.x_, self.y_, self.z_, test_size=0.2)

    def ready_data(self, N: int) -> tuple[np.array, np.array, np.array]:
        """Preprocess the terrain data

        inputs:
            N (int): number of points
        returns:
            Values from the meshgrid of points, with the corresponing height
        """

        width, length = self.terrainData.shape

        x_ = np.sort(np.linspace(0, 1, N + 1))
        y_ = np.sort(np.linspace(0, 1, N + 1))

        self.x_raw, self.y_raw = np.meshgrid(x_, y_)
        x = self.x_raw.flatten().reshape(-1, 1)
        y = self.y_raw.flatten().reshape(-1, 1)

        self.z_raw = self.terrainData[: 2 * N + 1 : 2, : 2 * N + 1 : 2]
        self.scaled_z = (self.z_raw - np.mean(self.z_raw)) / np.std(self.z_raw)
        t = self.scaled_z.flatten().reshape(-1, 1)

        return x, y, t

    def plotSurface(self) -> None:
        "Plot the surface of the data."
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection="3d")
        surf = ax.plot_surface(
            self.x_raw,
            self.y_raw,
            self.z_raw,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=True,
        )

        # Customize the z axis.
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        ax.set_title("Not scaled")

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=10)

        ax = fig.add_subplot(1, 2, 2, projection="3d")
        surf = ax.plot_surface(
            self.x_raw,
            self.y_raw,
            self.scaled_z,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=True,
        )

        # Customize the z axis.
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        ax.set_title("Scaled")

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=10)

        fig.suptitle("Terrain Data")

        plt.tight_layout()

        if self.savePlots:
            plt.savefig(self.figsPath / "TerrainFigure.png", dpi=300)
        if self.showPlots:
            plt.show()
        plt.close()
