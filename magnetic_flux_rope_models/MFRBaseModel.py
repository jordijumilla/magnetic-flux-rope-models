import abc
import numpy as np
import pandas as pd
import pickle
import math
import time
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
from scipy.optimize import fmin_l_bfgs_b

from magnetic_flux_rope_models.RandomNoise import RandomNoise, UniformNoise, GaussianNoise
from magnetic_flux_rope_models.OptimisationEngine import OptimisationParameter


class MFRBaseModel():
    """MFRBaseModel is a Python interface that defines the methods that all magnetic
    flux rope (MFR) models classes should have."""
    def __init__(self) -> None:
        # Physical constants.
        self.mu_0 = 4 * math.pi * (10 ** (-7))        
        self.AU_to_m = 149_597_870_700.0
    
    @abc.abstractmethod
    def __repr__(self) -> str:
        """Create nice string to display the parameters of the model to the user in a string format."""
        raise NotImplementedError

    @abc.abstractmethod
    def _validate_parameters(self) -> None:
        """Validate the parameters of the magnetic flux rope model."""
        raise NotImplementedError

    @abc.abstractmethod
    def simulate_crossing(self, *args, **kwargs) -> pd.DataFrame | None:
        """Simulate a crossing of a spacecraft through the magnetic flux rope."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def _validate_crossing_parameters(self, *agrs, **kwargs) -> None:
        """Validate the crossing parameters of the spacecraft through the magnetic flux rope."""
        raise NotImplementedError

    def get_noise_generator(self, noise_type: str, epsilon: float, random_seed: int = 0) -> RandomNoise:
        """Provide a random noise generator instance, of the desired type and with the desired parameters."""
        # Noise options.
        noise_type = noise_type.lower()
        self._validate_noise(noise_type, epsilon)

        # Instantiate the noise generator, depending on the noise type desired.
        if noise_type == "uniform":
            return UniformNoise(epsilon, random_seed=random_seed)

        return GaussianNoise(mu=0, sigma=epsilon, random_seed=random_seed)

    def _validate_noise(self, noise_type: str, epsilon: float) -> None:
        # Parameter: noise_type.
        if not isinstance(noise_type, str):
            raise ValueError("Parameter: noise_type must be string or None.")
        if noise_type != "uniform" and noise_type != "gaussian":
            raise ValueError("Parameter: noise_type must be of one of the supported string options: 'uniform' or 'gaussian'.")

        # Parameter: epsilon.
        if not (epsilon > 0):
            raise ValueError("Parameter: epsilon must be > 0.")
    
    @staticmethod
    def convert_units_magnetic_flux(magnitude, input_units: str, output_units: str):
        if input_units == output_units:
            # Do not need to change the units.
            return magnitude
        elif input_units == "Wb" and output_units == "Mx":
            return magnitude * 1e8
        elif input_units == "Mx" and output_units == "Wb":
            return magnitude / 1e8
        else:
            raise ValueError(f"Conversion from {input_units} to {output_units} is not supported.")

    @staticmethod
    def cartesian_vector_magnitude(v_x: np.ndarray, v_y: np.ndarray, v_z: np.ndarray) -> np.ndarray:
        return np.sqrt(np.square(v_x) + np.square(v_y) + np.square(v_z))
    
    @staticmethod
    def get_force_density_field_from_unit_right_handed_orthogonal_basis(J_field: np.ndarray, B_field: np.ndarray) -> np.ndarray:
        """Calculate the force density field (force per unit volume) from the Lorentz equation:
        
        F = rho_e * E + J x B
        
        Assuming that the electrical term is zero.
        Arguments are assumed to be expressed in a unit, right-handed & orthogonal vector basis.

        Args:
            J_field (np.ndarray): Current density field.
            B_field (np.ndarray): Magnetic field.

        Returns:
            np.ndarray: _description_
        """
        # Lorentz equation: F = J x B (assuming electric field E = 0).
        return np.cross(J_field, B_field)
    
    @staticmethod
    def _link_3D_axes_view(fig: Figure, ax: np.ndarray) -> callable:
        def on_move(event) -> None:
            """Callback function to move any number of figure axis together."""
            # Start by finding what axis is being moved.
            moved_axis = None
            moved_axis_idx = None
            bFound = False
            for ax_idx_0, ax_row in enumerate(ax):
                for ax_idx_1, axis in enumerate(ax_row):
                    if not bFound and event.inaxes == axis:
                        moved_axis = axis
                        moved_axis_idx = (ax_idx_0, ax_idx_1)
                        bFound = True

            if moved_axis is None:
                return

            # Apply the same movement to the rest of axis, except for the one that is already being moved.
            for ax_idx_0, ax_row in enumerate(ax):
                for ax_idx_1, axis in enumerate(ax_row):
                    if not (ax_idx_0 == moved_axis_idx[0] and ax_idx_1 == moved_axis_idx[1]):
                        axis.view_init(elev=moved_axis.elev, azim=moved_axis.azim)

            fig.canvas.draw_idle()

        return on_move

    @staticmethod
    def evaluate_model_and_crossing(model_class,
                                    df_observations: pd.DataFrame,
                                    residue_method: str,
                                    model_parameters: dict[str, float]  | None = None,
                                    crossing_parameters: dict[str, float] | None = None) -> tuple[float | None, None]:
        if model_parameters is None:
            model_parameters = dict()
        
        if crossing_parameters is None:
            crossing_parameters = dict()

        # Instantiate the EC Model with the given parameters.
        mfr_model = model_class(**model_parameters)

        # Use the same number of points for the fitting as the incoming observations.
        crossing_parameters["time_stencil"] = df_observations["time"].to_numpy()

        # Set this flag to avoid unnecessary computations like current density and force density.
        crossing_parameters["is_fitting"] = True

        # Simulate the crossing.
        df_test: pd.DataFrame | None = mfr_model.simulate_crossing(**crossing_parameters)

        if df_test is None:
            # There is no intersection.
            return 1e9, None

        if residue_method in ["MSE", "RMSE"]:
            residue = ((df_observations[["B_x", "B_y", "B_z"]] - df_test[["B_x", "B_y", "B_z"]])**2).to_numpy().sum()
        
            if residue_method == "RMSE":
                residue = math.sqrt(residue / len(df_observations))
            elif residue_method == "MSE":
                residue /= len(df_observations)
        
        elif residue_method == "SSE":
            residue = ((df_observations[["B_x", "B_y", "B_z"]] - df_test[["B_x", "B_y", "B_z"]])**2).to_numpy().sum()

        elif residue_method == "X":
            # Method used by Nieves-Chinchilla et al. (2017) in "Elliptic-cylindrical Analytical Flux Rope Model for Magnetic Clouds"
            B_tot_observations = np.sqrt(np.square(df_observations["B_x"]) + np.square(df_observations["B_y"]) + np.square(df_observations["B_z"]))
            B_tot_test = np.sqrt(np.square(df_test["B_x"]) + np.square(df_test["B_y"]) + np.square(df_test["B_z"]))
            residue += np.sum(np.square(B_tot_observations - B_tot_test))
            residue /= np.max(B_tot_observations)**2
            residue = math.sqrt(residue)
            residue /= len(df_observations)

        return residue, mfr_model
    
    @staticmethod
    def fit(model_class,
            df_observations: pd.DataFrame,
            model_parameters: dict[str, float],
            crossing_parameters: dict[str, float],
            residue_method: str = "RMSE",
            timeit: bool = False):
        # If the user wants timing information, start a time counter.
        if timeit:
            t1 = time.perf_counter()

        # Parse the model and crossing parameters
        model_parameters_parsed: list[OptimisationParameter] = [OptimisationParameter(name=parameter_name, options=parameter_options) for parameter_name, parameter_options in model_parameters.items()]
        crossing_parameters_parsed: list[OptimisationParameter] = [OptimisationParameter(name=parameter_name, options=parameter_options) for parameter_name, parameter_options in crossing_parameters.items()]

        # Split the parameters between optimised and fixed.
        model_parameters_to_optimise = [p for p in model_parameters_parsed if p.mode == "optimised"]
        model_parameters_fixed = {p.name: p.fixed_value for p in model_parameters_parsed if p.mode == "fixed"}

        crossing_parameters_to_optimise = [p for p in crossing_parameters_parsed if p.mode == "optimised"]
        crossing_parameters_fixed = {p.name: p.fixed_value for p in crossing_parameters_parsed if p.mode == "fixed"}

        def function_to_optimise(x: list[float]) -> float:
            # Assign each of the optimising variables to its corresponding parameter.
            model_kw_parameters = {p.name: x[idx] for idx, p in enumerate(model_parameters_to_optimise)} | model_parameters_fixed 

            n_offset: int = len(model_parameters_to_optimise)
            crossing_kw_parameters = {p.name: x[n_offset + idx] for idx, p in enumerate(crossing_parameters_to_optimise)} | crossing_parameters_fixed

            residue, _ = model_class.evaluate_model_and_crossing(model_class,
                                                                 df_observations,
                                                                 model_parameters=model_kw_parameters,
                                                                 crossing_parameters=crossing_kw_parameters,
                                                                 residue_method=residue_method)
            return residue
            
        bounds = [(p.bounds[0], p.bounds[1]) for p in model_parameters_to_optimise] + [(p.bounds[0], p.bounds[1]) for p in crossing_parameters_to_optimise]
        initial_parameters: list[float] = [p.initial_value for p in model_parameters_to_optimise] + [p.initial_value for p in crossing_parameters_to_optimise]

        # Call the L-BFGS-B optimiser.
        x_opt, f_opt, info = fmin_l_bfgs_b(func=function_to_optimise, x0=initial_parameters, factr=1e4, pgtol=1e-9, bounds=bounds, approx_grad=True)
        
        # Populate the info with the optimisation results, for debug purposes.
        info["x_opt"] = x_opt
        info["f_opt"] = f_opt
        info["function_calls"] = info["funcalls"]
        info["number_of_iterations"] = info["nit"]

        # Check if there is convergence.
        if info["warnflag"] != 0:
            # TODO: Restart the optimisation with a different initial guess.
            print(f"""Optimisation did not converge: {info["warnflag"]}.""")
            return None, None, None, None, info

        model_parameters_opt = {p.name: x_opt[idx] for idx, p in enumerate(model_parameters_to_optimise)}
        info["model_parameters_opt"] = model_parameters_opt

        model_parameters_all = model_parameters_opt | model_parameters_fixed
        n_offset: int = len(model_parameters_to_optimise)

        crossing_parameters_opt = {p.name: x_opt[n_offset + idx] for idx, p in enumerate(crossing_parameters_to_optimise)}
        info["crossing_parameters_opt"] = crossing_parameters_opt

        crossing_parameters_all = crossing_parameters_opt | crossing_parameters_fixed
        crossing_parameters_all["time_stencil"] = df_observations["time"].to_numpy()

        fitted_model = model_class(**model_parameters_all)

        # Output the fitted simulated dataset.
        fitted_df: pd.DataFrame = fitted_model.simulate_crossing(**crossing_parameters_all)

        if timeit:
            t2 = time.perf_counter()
            info["fitting_time"] = t2 - t1

        return fitted_model, model_parameters_all, crossing_parameters_all, fitted_df, info

    def compute_fitting_metrics(self, df_observations: pd.DataFrame, df_fitted: pd.DataFrame) -> dict[str, float]:
        """Compute fitting metrics RMSE and R^2 between observed and fitted data.

        Args:
            df_observations (pd.DataFrame): The observed data.
            df_fitted (pd.DataFrame): The fitted data.

        Returns:
            dict[str, float]: A dictionary containing the computed metrics.
        """
        # Initialise the metrics dictionary.
        metrics = dict()

        # Compute RMSE for each component and total.
        metrics["RMSE_x"] = math.sqrt(np.sum(np.square(df_observations["B_x"] - df_fitted["B_x"])) / len(df_observations))
        metrics["RMSE_y"] = math.sqrt(np.sum(np.square(df_observations["B_y"] - df_fitted["B_y"])) / len(df_observations))
        metrics["RMSE_z"] = math.sqrt(np.sum(np.square(df_observations["B_z"] - df_fitted["B_z"])) / len(df_observations))
        metrics["RMSE"] =  math.sqrt(metrics["RMSE_x"]**2 + metrics["RMSE_y"]**2 + metrics["RMSE_z"]**2)

        # Compute R^2 for each component and total.
        SS_res_x = np.sum(np.square(df_observations["B_x"] - df_fitted["B_x"]))
        SS_tot_x = np.sum(np.square(df_observations["B_x"] - np.mean(df_observations["B_x"])))
        metrics["R^2_x"] = float(1 - SS_res_x / SS_tot_x)

        SS_res_y = np.sum(np.square(df_observations["B_y"] - df_fitted["B_y"]))
        SS_tot_y = np.sum(np.square(df_observations["B_y"] - np.mean(df_observations["B_y"])))
        metrics["R^2_y"] = float(1 - SS_res_y / SS_tot_y)

        SS_res_z = np.sum(np.square(df_observations["B_z"] - df_fitted["B_z"]))
        SS_tot_z = np.sum(np.square(df_observations["B_z"] - np.mean(df_observations["B_z"])))
        metrics["R^2_z"] = float(1 - SS_res_z / SS_tot_z)

        SS_res_tot = SS_res_x + SS_res_y + SS_res_z
        SS_tot = SS_tot_x + SS_tot_y + SS_tot_z
        metrics["R^2"] = float(1 - SS_res_tot / SS_tot)
        return metrics

    def plot_vs_time(self,
                     data: pd.DataFrame,
                     magnitude_names: str | list[str],
                     colour: str | list[str],
                     time_units: str = "s",
                     datetime_axis: bool = False,
                     marker: str = "o",
                     linestyle: str = "-",
                     markersize: str = 4,
                     alpha: float = 1.0,
                     fig_size: tuple[float, float] | None = None,
                     ax: matplotlib.axis.Axis| None = None) -> None | matplotlib.axis.Axis:
        if isinstance(magnitude_names, str):
            magnitude_names = [magnitude_names]
        
        if isinstance(colour, str):
            colour = [colour]

        axis_provided: bool = ax is not None

        # Extract the time from the dataframe.
        if not datetime_axis:
            time = data["time"].to_numpy(copy=True)
            if time_units in {"min", "minute"}:
                time /= 60
            elif time_units in {"h", "hour"}:
                time /= 60 * 60
            elif time_units == "day":
                time /= 24 * 60 * 60
        else:
            time = data["datetime"]

        time_min: float = np.min(time)
        time_max: float = np.max(time)

        if not axis_provided:
            _, ax = plt.subplots(1, 1, tight_layout=True, figsize=fig_size)

        for idx, magnitude_name in enumerate(magnitude_names):
            # Extract the magnitude to plot from the dataframe, and its units.
            magnitude_to_plot = data[magnitude_name].to_numpy(copy=True)
            magnitude_units = self._units[magnitude_name]
            ax.plot(time, magnitude_to_plot, marker=marker, linestyle=linestyle, markersize=markersize, color=colour[idx], alpha=alpha)
            if datetime_axis:
                ax.set_xlabel("datetime")
            else:
                ax.set_xlabel(f"time ({time_units})")
            ax.set_ylabel(f"${magnitude_name}$ ({magnitude_units})")
            ax.grid(which="major", alpha=0.5)
            ax.grid(which="minor", alpha=0.25, linestyle=':')
            ax.minorticks_on()
            ax.set_xlim(time_min, time_max)
            
        plt.legend([f"${mag}$" for mag in magnitude_names])

        if not axis_provided:
            # If no axis was provided, show the plot.
            plt.show()
        else:
            # If an axis was provided, return it and do not show the plot.
            return ax

    def plot_crossing_magnetic_difference(self, df_to_fit: pd.DataFrame, df_fitted: pd.DataFrame, save_filename: str | None = None) -> None:
        """Plot the difference between the observed and fitted magnetic field components.

        Args:
            df_to_fit (pd.DataFrame): The observed data.
            df_fitted (pd.DataFrame): The fitted data.
        """
        _, ax = plt.subplots(figsize=(10, 6))
        self.plot_vs_time(df_to_fit, ["B_x", "B_y", "B_z", "B"], colour=["r", "g", "b", "k"], time_units="h", alpha=0.8, ax=ax)
        self.plot_vs_time(df_fitted, ["B_x", "B_y", "B_z", "B"], colour=["r", "g", "b", "k"], time_units="h", marker=None, linestyle="--", alpha=0.8, ax=ax)
        ax.set_title("Magnetic field components comparison vs time: simulated vs fitted")

        if save_filename is not None:
            plt.savefig(save_filename, bbox_inches="tight")
        else:
            plt.show()

    @staticmethod
    def _add_pickle_extension(file_path: str) -> str:
        """Add the pickle extension to the file path if it doesn't already contain it.

        Args:
            file_path (str): A path to a file.

        Returns:
            str: Same path, with the pickle extension if it didn't contain it.
        """
        if not file_path.endswith((".pkl", ".pickle")):
            return file_path + ".pkl"
        return file_path

    def save(self, file_path: str) -> None:
        """Save the MFR model in pickle format."""
        file_path_pickle: str = self._add_pickle_extension(file_path)
        
        with open(file_path_pickle, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(cls, file_path: str):
        """Load a MFR model in pickle format."""
        file_path_pickle: str = cls._add_pickle_extension(file_path)

        with open(file_path_pickle, "rb") as f:
            model = pickle.load(f)
        return model
