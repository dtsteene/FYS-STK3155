from FiniteDiffHeat import IterationMethod
import jax.numpy as np
from jax import jit, lax
from sklearn.metrics import mean_squared_error
from pathlib import Path
from plotutils import plot_surface, plot_at_timestepEuler


@jit
def analytic(x: np.ndarray, t: float):
    res = lax.mul(
        lax.exp(
            lax.mul(
                -lax.integer_pow(
                    np.pi,
                    2,
                ),
                t,
            )
        ),
        lax.sin(
            lax.mul(
                np.pi,
                x,
            )
        ),
    )
    return res
    # return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)


if __name__ == "__main__":
    figures_dir = Path(__file__).parent.parent / "figures/1dHeat"
    figures_dir.mkdir(exist_ok=True, parents=True)

    dxs = [1e-1, 1e-3]
    L = 1
    T = 0.5
    nSavePoints = 2

    for dx in dxs:
        dt = 0.5 * dx**2  # stability constraint

        x_arr = np.linspace(0, L, int(L / dx))
        x_anal = np.linspace(0, 1, 1000)
        Nt = int(T / dt)

        IM = IterationMethod(dx, dt, nSavePoints=nSavePoints, T=T, L=L)
        u_numerical, solve_times = IM.solve()
        u_numerical = u_numerical.T

        T_, X_ = np.meshgrid(solve_times, x_anal)

        u_analytical = analytic(X_.flatten(), T_.flatten())
        u_analytical = u_analytical.reshape(X_.shape)

        indicies = [0, int(Nt / 2), Nt - 1]

        for index in indicies:
            timestep = solve_times[index]
            res_numerical = u_numerical[:, index]
            res_analytic = u_analytical[:, index]
            plot_at_timestepEuler(
                x_arr,
                res_numerical,
                x_anal,
                res_analytic,
                timestep,
                "Forward Euler",
                True,
                figures_dir,
                f"difference_surf_dx_{dx}",
            )

        """
        def plot_at_timestep(
        x: np.ndarray,
        res_dnn: np.ndarray,
        res_analytic: np.ndarray,
        t: float,
        func_name: str,
        save: bool,
        savePath: Path = None,
        saveName: str = None,
        ) -> None:
        """

        """

        T_, X_ = np.meshgrid(solve_times, x_arr)

        u_analytical = analytic(X_.flatten(), T_.flatten())
        u_analytical = u_analytical.reshape(X_.shape)

        u_difference = u_analytical - u_numerical
        mse = mean_squared_error(u_analytical, u_numerical)
        plot_surface(
            T_,
            X_,
            u_numerical,
            "Numerical solution",
            True,
            figures_dir,
            f"numerical_surf_dx_{dx}",
        )
        plot_surface(
            T_,
            X_,
            u_difference,
            "Difference",
            True,
            figures_dir,
            f"difference_surf_dx_{dx}",
        )
        """
