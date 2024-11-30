import numpy as np
import pandas as pd
import math
from scipy.special import jv
from MagneticFluxRopeModels.EllipticalCylindricalModel import EllipticalCylindricalModel


class LundquistModel(EllipticalCylindricalModel):
    """Implementation of the Lundquist model, which is a circular-cylindrical symmetrical magnetic flux rope model.
    
    Magnetic field components:
        B_r(r) = 0
        B_phi(r) = B_z_0 * J_1(alpha * r / R)
        B_z(r) = B_z_0 * J_0(alpha * r / R)

    where J_0 and J_1 are the Bessel functions of first kind and zero-th and first order, respectively.

    The Lundquist model is a force-free model.
    """
    def __init__(
        self,
        R: float = 0.05,
        alpha: float = 2.0,
        B_z_0: float = 10.0,
        handedness: int = 1,
    ) -> None:
        """Initialise a Lundquist circular-cylindrical model.

        Args:
            R (float, optional): Radius of the magnetic flux rope. Defaults to 0.05.
            alpha (float, optional): Magnetic field scale parameter.. Defaults to 2.0.
            B_z_0 (float, optional): Axial magnetic field in the centre (in nT). Defaults to 10.0.
            handedness (int, optional): Positive handedness (+1) or negative handedness (-1). Defaults to +1.
        """
        # Initialise the EllipticalCylindricalModel superclass.
        super().__init__(delta=1.0, R=R, psi=0.0)
        

        # LundquistModel field parameters:
        self.alpha: float = alpha
        self.B_z_0: float = B_z_0
        self.handedness: int = handedness

        # Validate the incoming user-defined parameters.
        self._validate_parameters()

        # Create a dictionary with the units used for each magnitude.
        self._units = {
            "B": "nT",
            "B_x": "nT",
            "B_y": "nT",
            "B_z": "nT",
            "J": "pA",
            "J_x": "pA",
            "J_y": "pA",
            "J_z": "pA",
            "time": "s",
            "J": "pA",
            "pressure": "nPa",
        }

    def _validate_parameters(self) -> None:
        """Protected auxiliary function used to validate the input parameters to the model. This function is not meant to be called
        from outside of the class."""
        # Start by validating the elliptical-cylindrical symmetrical model parameters.
        super()._validate_elliptical_cylindrical_parameters()

        # Parameter: alpha.
        if not isinstance(self.alpha, (int, float)):
            raise TypeError("Parameter: alpha must be an integer or float.")

        if not (self.alpha > 0):
            raise ValueError("Parameter: alpha must be > 0.")

        # Parameter: handedness.
        if type(self.handedness) is not int:
            raise TypeError("Parameter: handedness must be an integer.")

        if self.handedness not in [-1, 1]:
            raise ValueError("Parameter: handedness must be in {-1, 1}")


    def __repr__(self) -> str:
        """Create nice string to display the parameters of the model to the user in a string format."""
        
        return f"""Lundquist model with parameters:
        - Geometrical:
            - R = {self.R:.2f} AU
        - Field:
            - B_z_0 = {self.B_z_0} nT
            - handedness = {self.handedness}."""

    def radial_argument(self, r: float) -> float:
        """Calculate the radial argument that appears in the Lundquist model, that is, alpha * (r / R).

        Args:
            r (float): Radial coordinate.

        Returns:
            float: alpha * (r / R)
        """
        return self.alpha * r / self.R

    def get_magnetic_field_elliptical_coordinates(self, r: float, phi: float) -> np.ndarray:
        """Calculate the magnetic field in the elliptical vector basis."""
        B_r: float = 0
        radial_argument: float = self.radial_argument(r=r)
        B_phi: float = self.handedness * self.B_z_0 * jv(1, radial_argument)
        B_z: float = self.B_z_0 * jv(0, radial_argument)

        return np.array([B_r, B_phi, B_z])

    def get_current_density_field_elliptical_coordinates(self, r: float, phi: float) -> np.ndarray:
        """Calculate the current density J in elliptical coordinates.

        Args:
            r (float): Radial coordinate.
            phi (float): Angular coordinate.

        Returns:
            np.ndarray: Array with [J_r, J_phi, J_z].
        """
        # The radial current density component is zero.
        J_r: float = 0

        # Pre-compute some magnitudes to avoid repeated code and improve performance.
        radial_argument: float = self.radial_argument(r=r)
        bessel_0: float = jv(0, radial_argument)
        bessel_1: float = jv(1, radial_argument)
        bessel_2: float = jv(2, radial_argument)
        alpha_over_R: float = self.alpha / self.R

        # TODO: Review this calculation.
        J_phi: float = self.B_z_0 * bessel_1 * alpha_over_R
        J_z: float = (self.handedness * self.B_z_0 / r) * (bessel_1 + 0.5 * r * alpha_over_R * (bessel_0 - bessel_2))

        return np.array([J_r, J_phi, J_z])

    def get_force_density_field_elliptical_coordinates(self, r: float | None = None, phi: float | None = None, J_field: np.ndarray | None = None, B_field: np.ndarray | None = None) -> np.ndarray:
        """Get the force field coordinates of the Lundquist model. Given that it is a force free model, the force field is null.

        Args:
            r (float): Radial coordinate.
            phi (float): Angular coordinate.

        Returns:
            np.ndarray: An array of [0, 0, 0].
        """
        if (isinstance(r, (int, float)) and isinstance(phi, (int, float))) or (J_field.ndim == 1 and B_field.ndim == 1):
            return np.array([0, 0, 0])
        
        if (isinstance(r, np.ndarray) and isinstance(phi, np.ndarray) and r.shape == phi.shape)or (J_field.ndim == 2 and B_field.ndim == 2):
            length = len(r) if isinstance(r, np.ndarray) else J_field.shape[0]
            return np.zeros((length, 3))
        
        raise ValueError(f"Invalid input types or shapes of r ({type(r)}) and phi ({type(phi)}).")


def main() -> None:
    my_lundquist_model = LundquistModel()
    my_lundquist_model.plot_boundary()
    my_lundquist_model.radial_coordinate_sweep(plot=True)
    my_lundquist_model.radial_and_angular_sweep(plot=True)
    df = my_lundquist_model.simulate_crossing(y_0=0.0, v_sc=450)
    my_lundquist_model.plot_vs_time(df, ["B_x", "B_y", "B_z", "B"], colour=["r", "g", "b", "k"], time_units="h")


if __name__ == "__main__":
    main()
