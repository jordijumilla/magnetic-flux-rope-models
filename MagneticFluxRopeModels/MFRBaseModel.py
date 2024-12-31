import abc
import numpy as np
import pandas as pd
import pickle
import math
import time
from matplotlib.figure import Figure
from typing import Self
from scipy.optimize import fmin_l_bfgs_b

from MagneticFluxRopeModels.RandomNoise import RandomNoise, UniformNoise, GaussianNoise
from MagneticFluxRopeModels.OptimisationEngine import OptimisationParameter


class MFRBaseModel():
    """MFRBaseModel is a Python interface that defines the methods that all magnetic
    flux rope (MFR) models classes should have."""
    def __init__(self) -> None:
        # Physical constants and unit conversions.
        self.mu_0 = 4 * math.pi * (10 ** (-7))        
        self.AU_to_m = 149_597_870_700.0

    # @classmethod
    # def __subclasshook__(cls, subclass):
    #     return (
    #         hasattr(subclass, "__init__")
    #         and callable(subclass.__init__)
    #         and hasattr(subclass, "_validate_parameters")
    #         and callable(subclass._validate_parameters)
    #         or NotImplemented
    #     )
    
    @abc.abstractmethod
    def __repr__(self) -> str:
        """Create nice string to display the parameters of the model to the user in a string format."""
        raise NotImplementedError

    @abc.abstractmethod
    def _validate_parameters(self) -> None:
        """Validate the parameters of the magnetic flux rope model."""
        raise NotImplementedError

    @abc.abstractmethod
    def simulate_crossing(self, *agrs, **kwargs):
        """Simulate a crossing of a spacecraft through the magnetic flux rope."""
        raise NotImplementedError
    
    @abc.abstractmethod
    def _validate_crossing_parameters(self, *agrs, **kwargs):
        """Validate the crossing parameters of the spacecraft through the magnetic flux rope."""
        raise NotImplementedError

    def get_noise_generator(self, noise_type: str, epsilon: float) -> RandomNoise:
        """Provide a random noise generator instance, of the desired type and with the desired parameters."""
        # Noise options.
        noise_type = noise_type.lower()
        self._validate_noise(noise_type, epsilon)

        # Instantiate the noise generator, depending on the noise type desired.
        if noise_type == "uniform":
            return UniformNoise(epsilon)

        return GaussianNoise(mu=0, sigma=epsilon)

    def _validate_noise(self, noise_type: str, epsilon: float) -> None:
        # Parameter: noise_type.
        if not isinstance(noise_type, str):
            raise ValueError("Parameter: noise_type must be string or None.")
        if noise_type != "uniform" and noise_type != "gaussian":
            raise ValueError(
                "Parameter: noise_type must be of one of the supported string options: 'uniform' or 'gaussian'."
            )

        # Parameter: epsilon.
        if not (epsilon > 0):
            raise ValueError("Parameter: epsilon must be > 0.")
        
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
    def evaluate_model_and_crossing(model_class: Self,
                                    df_observations: pd.DataFrame,
                                    residue_method: str,
                                    model_parameters: dict[str, float | None] = {},
                                    crossing_parameters: dict[str, float | None] = {},
                                    matching_manitudes: list[str] = []) -> tuple[float | None, Self | None]:
        # delta: float, psi: float, n: int, m: int, v_sc: float, y_0: float
        # Instantiate the EC Model with the given parameters.
        mfr_model = model_class(**model_parameters)

        # Use the same number of points for the fitting as the incoming observations.
        crossing_parameters["num_points"] = len(df_observations)

        # Simulate the crossing.
        df_test: pd.DataFrame | None = mfr_model.simulate_crossing(**crossing_parameters)

        if df_test is None:
            # There is no intersection.
            return 1e9, None

        if residue_method in ["SE", "MSE", "RMSE"]:
            residue = np.sum(np.square(df_observations["B_x"] - df_test["B_x"]))
            residue += np.sum(np.square(df_observations["B_y"] - df_test["B_y"]))
            residue += np.sum(np.square(df_observations["B_z"] - df_test["B_z"]))

        if residue_method in ["MSE", "RMSE"]:
            residue /= len(df_observations)
        
        if residue_method == "RMSE":
            residue = math.sqrt(residue)

        return residue, mfr_model
    
    @staticmethod
    def fit(model_class: Self,
            df_observations: pd.DataFrame,
            model_parameters: dict,
            crossing_parameters: dict,
            residue_method: str = "MSE",
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

        # Call the optimiser.
        x_opt, f_opt, info = fmin_l_bfgs_b(func=function_to_optimise, x0=initial_parameters, pgtol=1e-9, bounds=bounds, approx_grad=True)
        
        # Populate the info with the optimisation results, for debug purposes.
        info["x_opt"] = x_opt
        info["f_opt"] = f_opt

        # Check if there is convergence.
        if info["warnflag"] != 0:
            print(f"Optimisation did not converge: {info["warnflag"]}.")
            return None, None, info

        model_parameters_opt = {p.name: x_opt[idx] for idx, p in enumerate(model_parameters_to_optimise)} | model_parameters_fixed
        n_offset: int = len(model_parameters_to_optimise)
        crossing_parameters_opt = {p.name: x_opt[n_offset + idx] for idx, p in enumerate(crossing_parameters_to_optimise)}| crossing_parameters_fixed
        crossing_parameters_opt["num_points"] = len(df_observations)

        fitted_model = model_class(**model_parameters_opt)

        # Output the fitted simulated dataset.
        fitted_df: pd.DataFrame = fitted_model.simulate_crossing(**crossing_parameters_opt)

        if timeit:
            t2 = time.perf_counter()
            info["fitting_time"] = t2 - t1

        return fitted_model, crossing_parameters_opt, fitted_df, info

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


