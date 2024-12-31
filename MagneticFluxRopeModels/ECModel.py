# export PYTHONPATH=$PYTHONPATH:/Users/macbookpro/Documents/grad-shafranov-reconstruction-technique/

import numpy as np
import pandas as pd
import math
from MagneticFluxRopeModels.EllipticalCylindricalModel import EllipticalCylindricalModel

# TODOs:
# Add multiple n & m terms in the model.
# Add inclination angle in the crossing.

class ECModel(EllipticalCylindricalModel):
    """
    The Elliptic-Cylindrical (EC) Magnetic Flux Rope Model.

    A Python implementation of the model by Teresa Nieves-Chinchilla et al. from https://doi.org/10.3847/1538-4357/aac951

    Parameters
    ----------
    All angles are expected in units of degrees.

    delta: float, A measure of the ellipticity of the flux rope.
           The ratio between the length of the minor and major axes
           of the elliptical cross section. Valid range (0, 1].

    psi: float, Angle of rotation about the central axis of the flux rope.
        Valid range [0, 180].

    handedness: int, optional {-1, 1} Handedness of flux rope.

    C_nm: int, optional, default: 1.0
        Valid only if > 0

    B_z_0: float, optional, default: 10.0 nT

    tau: float, optional, default: 1.3

    R: float, optional, default: 0.05 (diameter = 0.1)
        Radius of the semi-major axis in Astronomical Units (AU)

    n: int, optional, default: 1

    m: int, optional, default: 0

    epsilon: float > 0, optional, default: 0.05
        Size of noise modifier. Ignored if noise_type is 'none'.

        For noise_type 'gaussian', epsilon is the standard deviation of the
        normal distrubution centered on 0. Values in the distribution
        Normal(mu=0,sigma=epsilon) are added to the magnetic field components.

        For noise_type 'uniform', epsilon defines the +/- bounds of the
        distribution.  Values in [-epsilon,epsilon] are added to the magnetic
        field components.
    """

    def __init__(
        self,
        delta: float,
        psi: float,
        R: float = 0.05,
        n: int = 1,
        m: int = 0,
        C_nm: float = 1.0,
        tau: float = 1.3,
        B_z_0: float = 10.0,
        handedness: int = 1,
        p_0: float = 0.0
    ) -> None:
        # Initialise the EllipticalCylindricalModel superclass.
        super().__init__(delta=delta, R=R, psi=psi)

        # EC Model field parameters.
        self.n: int = n
        self.m: int = m
        self.tau: float = tau
        self.C_nm: float = C_nm
        self.B_z_0: float = B_z_0
        self.handedness: int = handedness
        self.p_0: float = p_0

        # Validate the incoming user-defined parameters.
        self._validate_parameters()

        # Pre-compute alpha_n and beta_m parameters.
        self.R_to_n_plus_one: float = math.pow(self.R, self.n + 1)
        self.alpha_n: float = self.B_z_0 * (self.n + 1) / (self.mu_0 * self.delta * self.tau * self.R_to_n_plus_one)
        self.beta_m: float = self.B_z_0 * (self.n + 1) / (self.mu_0 * self.delta * self.C_nm * self.tau * math.pow(self.R, self.m + 1))

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
        if type(self.m) is not int:
            raise TypeError("Parameter: m must be an integer.")

        if self.m < 0:
            raise ValueError("Parameter: m must be >= 0.")

        # Parameter: handedness.
        if type(self.handedness) is not int:
            raise TypeError("Parameter: handedness must be an integer.")

        if self.handedness not in [-1, 1]:
            raise ValueError("Parameter: handedness must be in {-1, 1}")

        # Parameter: C_nm.
        if not isinstance(self.C_nm, (int, float)):
            raise TypeError("Parameter: C_nm must be an integer or float.")

        if self.C_nm <= 0:
            raise ValueError("Parameter: C_nm must be > 0.")

        # Parameter: p_0.
        if not isinstance(self.p_0, (int, float)):
            raise TypeError("Parameter: p_0 must be an integer or float.")

        if self.p_0 < 0:
            raise ValueError("Parameter: p_0 must be >= 0.")

    def __repr__(self) -> str:
        """Create nice string to display the parameters of the model to the user in a string format."""
        return f"""ECModel with parameters:
        - Geometrical:
            - delta = {self.delta:.3f}
            - psi = {math.degrees(self.psi):.3f} deg
            - R = {self.R:.3f} AU
        - Field:
            - n = {self.n}
            - m = {self.m}
            - tau = {self.tau:.3f}
            - C_nm = {self.C_nm:.3f}
            - B_z_0 = {self.B_z_0:.3f} nT
            - handedness = {self.handedness}."""

    def get_magnetic_field_elliptical_coordinates(self, r: float, phi: float) -> np.ndarray:
        """Calculate the magnetic field in the elliptical vector basis."""
        B_r: float = 0
        B_phi: float = (
            self.handedness
            * self.mu_0
            * self.get_h(phi + self.psi)
            * self.delta
            * self.beta_m
            * math.pow(r, self.m + 1)
            / (self.delta_squared + self.m + 1)
        )
        B_z: float = (
            self.mu_0
            * self.delta
            * self.alpha_n
            * (self.tau * self.R_to_n_plus_one - math.pow(r, self.n + 1))
            / (self.n + 1)
        )

        return np.array([B_r, B_phi, B_z])

    def get_current_density_field_elliptical_coordinates(self, r: float, phi: float) -> np.ndarray:
        # The radial current density component is zero.
        J_r: float = 0

        # Pre-compute h and chi factors.
        h: float = self.get_h(phi=phi + self.psi)
        chi: float = self.get_chi(phi=phi + self.psi)

        # Compute the poloidal and axial components of the current density, as per equation 30 from the paper.
        J_phi: float = h * self.alpha_n * math.pow(r, self.n)
        J_z: float = h * h * self.beta_m * (chi + self.m) * math.pow(r, self.m) / (self.delta_squared + self.m + 1)

        return np.array([J_r, J_phi, J_z]) / 1e9


def main() -> None:
    my_ec_model = ECModel(delta=0.7, psi=0.0)
    df = my_ec_model.simulate_crossing(v_sc=450, y_0=0)
    print(df)
    return
    # my_ec_model.radial_coordinate_sweep(two_fold=True, plot=True)

    # my_ec_model.simulate_crossing(v_sc=450, y_0=0.999)

    # Make a 2D sweep (radial & angular).
    # my_ec_model.radial_and_angular_sweep(plot=True)

    # my_ec_model.plot_boundary(normalise_radial_coordinate=True)
    #print(my_ec_model)

    # Make a radial sweep.
    # my_ec_model.radial_coordinate_sweep(plot=True, normalise_radial_coordinate=True)
    for noise_level in [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
        print(f"Noise level = {noise_level:.3f}.")
        n_delta = 9
        n_y_0 = 9
        results = np.zeros((n_delta*n_y_0, 4))
        idx = 0
        for delta in np.linspace(0.45, 1.0, n_delta, endpoint=True):
            for y_0 in np.linspace(0.0, 0.9, n_y_0, endpoint=True):
                my_ec_model = ECModel(delta=delta, psi=0.0) # , psi=0.0, R=0.05
                df = my_ec_model.simulate_crossing(v_sc=450.0, y_0=y_0, noise_type="gaussian", epsilon=noise_level*my_ec_model.B_z_0)
                result = ECModel.fit(ECModel, df)
                if result is not None:
                    fitted_model, y_0_opt, optimisation_result = result
                    fitted_deta = fitted_model.delta
                    print(f"Delta = {delta:.3f}, y_0 = {y_0:.3f} ---> Opt. delta = {fitted_deta:.3f}, Opt. y_0 = {y_0_opt:.3f}")
                    results[idx, :] = np.array([delta, y_0, fitted_deta, y_0_opt])
                else:
                    results[idx, :] = np.array([delta, y_0, 1e9, 1e9])
                idx += 1
                
        
        df_res = pd.DataFrame(results, columns=["delta", "y_0", "delta_opt", "y_0_opt"])
        df_res.to_csv(f"sim_results_{noise_level:.3f}.csv", index=False)
        # print(df_res)
    
    return
    print(df)
    # my_ec_model.plot_boundary(normalise_radial_coordinate=True)
    # return
    # print(my_ec_model)

    # # Make a radial sweep.
    # my_ec_model.radial_coordinate_sweep(bPlot=True, normalise_radial_coordinate=True)

    # # Make a 2D sweep (radial & angular).
    # my_ec_model.radial_and_angular_sweep(bPlot=True)

    # point = my_ec_model.convert_elliptical_to_cartesian_cordinates(r=my_ec_model.R, phi=phi, z=0)
    # basis = my_ec_model.get_elliptical_unit_basis(r=my_ec_model.R, phi=phi)

    # my_ec_model.plot_boundary(
    #     vector_dict={"$\hat{e}_r$": (point, basis[:, 0]), "$\hat{e}_{\phi}$": (point, basis[:, 1])}
    # )

    my_ec_model.plot_vs_time(df, ["B_x", "B_y", "B_z", "B"], colour=["r", "g", "b", "k"], time_units="h")
    my_ec_model.plot_vs_time(df, ["J_x", "J_y", "J_z", "J"], colour=["r", "g", "b", "k"], time_units="h")
    return

    # print(basis)

    # print(f"{basis[:, 0]=}")
    # scale_factors = my_ec_model.compute_scale_factors(r=my_ec_model.R, phi=phi)
    # print(f"{scale_factors=}")


    # my_data = my_ec_model.generate_data()
    # print(my_data)
    # my_ec_model.plot_vs_time(["B", "J"], time_units="day")


if __name__ == "__main__":
    main()
