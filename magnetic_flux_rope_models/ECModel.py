import numpy as np
from numpy.typing import NDArray
import math
from magnetic_flux_rope_models.EllipticalCylindricalModel import EllipticalCylindricalModel

# TODOs:
# Add multiple n & m terms in the model.
# Add inclination angle in the crossing.

class ECModel(EllipticalCylindricalModel):
    """The Elliptic-Cylindrical (EC) Magnetic Flux Rope Model.
    A Python implementation of the model by Teresa Nieves-Chinchilla et al. from https://doi.org/10.3847/1538-4357/aac951
    """

    def __init__(
        self,
        delta: float,
        xi: float,
        R: float = 0.05,
        n: int = 1,
        m: int = 0,
        C_nm: float = 1.0,
        tau: float = 1.3,
        B_z_0: float = 10.0,
        handedness: int = 1,
    ) -> None:
        """
        Initialise an ECModel instance.

        Args:
            delta (float): Ellipticity of the flux rope. Ratio between the lengths of the minor and major axes of the elliptical cross section. Valid range: (0, 1].
            xi (float): Angle of rotation about the central axis of the flux rope, in radians. Valid range: [0, π].
            R (float, optional): Radius of the semi-major axis in Astronomical Units (AU). Default is 0.05.
            n (int, optional): Model parameter n. Default is 1.
            m (int, optional): Model parameter m. Default is 0.
            C_nm (float, optional): Model parameter C_nm. Must be > 0. Default is 1.0.
            tau (float, optional): Model parameter tau. Default is 1.3.
            B_z_0 (float, optional): Central axial magnetic field strength in nT. Default is 10.0.
            handedness (int, optional): Handedness of the flux rope. Must be -1 or 1. Default is 1.
        """
        # Initialise the EllipticalCylindricalModel superclass.
        super().__init__(delta=delta, R=R, xi=xi)

        # EC Model field parameters.
        self.n: int = n
        self.m: int = m
        self.tau: float = tau
        self.C_nm: float = C_nm
        self.B_z_0: float = B_z_0
        self.handedness: int = handedness

        # Validate the incoming user-defined parameters.
        self._validate_parameters()

        # Pre-compute alpha_n and beta_m parameters.
        self.alpha_n: float = self.B_z_0 * (self.n + 1) / (self.mu_0 * self.delta * self.tau * math.pow(self.R * self.AU_to_m, self.n + 1))
        #self.beta_m: float = self.alpha_n * math.pow(self.R * self.AU_to_m, self.n - self.m) / C_nm
        self.beta_m: float = self.B_z_0 * (self.n + 1) / (self.mu_0 * self.delta * self.C_nm * self.tau * math.pow(self.R * self.AU_to_m, self.m + 1))

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
            "pressure": "nPa",
        }

    def _validate_parameters(self) -> None:
        """Protected auxiliary function used to validate the input parameters to the model. This function is not meant to be called
        from outside of the class."""
        # Start by validating the elliptical-cylindrical symmetrical model parameters.
        super()._validate_elliptical_cylindrical_parameters()

        # Parameter: n.
        if type(self.n) is not int:
            raise TypeError("Parameter: n must be an integer.")

        if self.n <= 0:
            raise ValueError("Parameter: n must be >= 1.")

        # Parameter: m.
        if not isinstance(self.m, int):
            raise TypeError("Parameter: m must be an integer.")

        if self.m < 0:
            raise ValueError("Parameter: m must be >= 0.")

        # Parameter: handedness.
        if not isinstance(self.handedness, int):
            raise TypeError("Parameter: handedness must be an integer.")

        if self.handedness not in [-1, 1]:
            raise ValueError("Parameter: handedness must be in {-1, 1}")

        # Parameter: C_nm.
        if not isinstance(self.C_nm, (int, float)):
            raise TypeError("Parameter: C_nm must be an integer or float.")

        if self.C_nm <= 0:
            raise ValueError("Parameter: C_nm must be > 0.")

    def __repr__(self) -> str:
        """Create nice string to display the parameters of the model to the user in a string format."""
        return f"""ECModel with parameters:
        - Geometrical:
            - delta = {self.delta:.3f}
            - xi = {math.degrees(self.xi):.3f} deg
            - R = {self.R:.3f} AU
        - Field:
            - n = {self.n}
            - m = {self.m}
            - tau = {self.tau:.3f}
            - C_{self.n}{self.m} = {self.C_nm:.3f}
            - B_z_0 = {self.B_z_0:.3f} nT
            - handedness = {self.handedness}."""

    def get_magnetic_field_elliptical_coordinates(self, r: float, phi: float) -> NDArray[np.float64]:
        """Calculate the magnetic field in the elliptical vector basis."""
        B_r: float = 0

        B_phi: float = (
            -self.handedness
            * self.mu_0
            * self.get_h(phi)
            * self.delta
            * self.beta_m
            * math.pow(r  * self.AU_to_m, self.m + 1)
            / (self.delta_squared + self.m + 1)
        )

        B_z: float = (
            self.B_z_0 -
            self.mu_0
            * self.delta
            * self.alpha_n
            * math.pow(r * self.AU_to_m, self.n + 1)
            / (self.n + 1)
        )

        return np.array([B_r, B_phi, B_z])

    def get_current_density_field_elliptical_coordinates(self, r: float, phi: float) -> NDArray[np.float64]:
        J_r: float = 0

        # Pre-compute h and chi factors.
        h: float = self.get_h(phi=phi)
        chi: float = self.get_chi(phi=phi)

        # Compute the poloidal and axial components of the current density, as per equation 30 from the paper.
        J_phi: float = h * self.alpha_n * math.pow(r * self.AU_to_m, self.n)
        J_z: float = h * h * self.beta_m * (chi + self.m) * math.pow(r * self.AU_to_m, self.m) / (self.delta_squared + self.m + 1)

        return np.array([J_r, J_phi, J_z]) / 1e9

    def get_twist(self, r: float, phi: float, L: float | None = None) -> float:
        B_field_elliptical = self.get_magnetic_field_elliptical_coordinates(r=r, phi=phi)

        # Convert the magnetic field from elliptical to cartesian coordinates
        B_field = self.convert_elliptical_to_cartesian_vector(B_field_elliptical[0], B_field_elliptical[1], B_field_elliptical[2], r, phi)
        B_phi = B_field[1]
        B_z = B_field[2]

        if L is None:
            L = 2 * math.pi * self.R

        return L*B_phi / (2*math.pi*r*B_z)
    
        # TODO: Check if this is correct.

        sin_phi: float = math.sin(phi)
        cos_phi: float = math.cos(phi)

        # Note that this factor is very similar to "h", but has the sine and cosine swapped.
        factor = math.sqrt(self.delta_squared * cos_phi * cos_phi + sin_phi * sin_phi)

        curl_z_B = 1/(r * factor)
        curl_z_B *= B_phi
        # curl_z_B *= r * # Partial B_phi w.r.t. the radial coordinate (r).

        twist = curl_z_B / B_z
        return twist
    
    def get_total_axial_magnetic_flux(self, units: str = "Wb") -> float:
        """Calculate the axial magnetic flux of the flux rope."""
        phi_z = self.mu_0 * math.pi * self.delta_squared * self.alpha_n * math.pow(self.R * self.AU_to_m, self.n + 3) * (self.tau - 2/(self.n + 3)) / (self.n + 1)
        
        # Convert units to SI.
        phi_z /= 1e9
        
        return self.convert_units_magnetic_flux(phi_z, input_units="Wb", output_units=units)

    def get_total_poloidal_magnetic_flux(self, units: str = "Wb") -> float:
        """Calculate the poloidal magnetic flux of the flux rope per unit length."""
        phi_poloidal = self.mu_0 * self.delta_squared * self.beta_m * math.pow(self.R * self.AU_to_m, self.m + 2)  / ((self.delta_squared * self.m + 1)*(self.m + 2))
        
        # Convert units to SI.
        phi_poloidal /= 1e9

        return self.convert_units_magnetic_flux(phi_poloidal, input_units="Wb", output_units=units)
