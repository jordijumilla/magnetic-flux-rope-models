import abc
import math
import matplotlib
import matplotlib.axis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from MagneticFluxRopeModels.MFRBaseModel import MFRBaseModel
from MagneticFluxRopeModels.RandomNoise import RandomNoise
import datetime


class EllipticalCylindricalModel(MFRBaseModel):
    """Common interface for ellipical-cylindrical symmetric MFR models. This includes circular-cylindrical models."""

    def __init__(self, delta: float, R: float, psi: float) -> None:
        super().__init__()

        # Geometrical.
        self.delta: float = delta
        self.R: float = R
        self.psi: float = psi

        # Validate these parameters
        self._validate_elliptical_cylindrical_parameters()

        # Calculate the derived geometrical magnitudes.
        self._calculate_derived_parameters()

    def _validate_elliptical_cylindrical_parameters(self) -> None:
        # Parameter: delta.
        if not isinstance(self.delta, (int, float)):
            raise TypeError("Parameter: delta must be an integer or float.")

        if not (0 < self.delta <= 1):
            raise ValueError("Parameter: delta must be in (0,1].")
        
        # Parameter: R.
        if not isinstance(self.R, (int, float)):
            raise TypeError("Parameter: R must be an integer or float.")
        
        if not (self.R > 0):
            raise ValueError("Parameter: R must be > 0.")

        # Parameter: psi.
        if not isinstance(self.psi, (int, float)):
            raise TypeError("Parameter: psi must be an integer or float.")

        if not (0 <= self.psi < math.pi):
            raise ValueError("Parameter: psi must be in [0, pi).")

    def _calculate_derived_parameters(self):
        # Delta-related magnitudes:
        self.delta_squared = self.delta * self.delta

        # Psi-related magnitudes:
        self.sin_psi: float = math.sin(self.psi)
        self.cos_psi: float = math.cos(self.psi)
        self.psi_rotation_matrix: np.ndarray = np.array([[self.cos_psi, -self.sin_psi, 0],
                                                         [self.sin_psi, self.cos_psi,  0],
                                                         [0,            0,             1]])


    def get_elliptical_unit_basis(self, r: float | np.ndarray, phi: float | np.ndarray) -> np.ndarray:
        """Calculate e_r and e_phi, given by the coordinate change:
            x = delta * r * cos(phi + psi)
            y = r * sin(phi + psi)
            z = z
        By differentating the previous change coordinate change w.r.t. r, phi and z, we can get
        the unit vectors in the elliptical basis. This basis is a non-orthogonal and non-unit vector basis.
        {epsilon_r, epsilon_phi} lay in the xy-plane and are thus both orthogonal to epsilon_z. However,
        they are not orthogonal between them. {epsilon_r, epsilon_phi} are non-unit, whereas epsilon_z has magnitude 1."""

        if isinstance(r, (int, float)) and isinstance(phi, (int, float)):
            epsilon_r: np.ndarray = [self.delta * math.cos(phi), math.sin(phi), 0]

            # This vector should be scaled by r, but because we are normalising it later, we avoid the multiplication by r.
            epsilon_phi: np.ndarray = np.array([-self.delta * math.sin(phi), math.cos(phi), 0])
            epsilon_z: np.ndarray = np.array([0, 0, 1])

            # Normalise the vectors.
            epsilon_r /= np.linalg.norm(epsilon_r)
            epsilon_phi /= np.linalg.norm(epsilon_phi)

            basis = np.stack([epsilon_r, epsilon_phi, epsilon_z]).T
            return self.psi_rotation_matrix @ basis
        
        else:
            epsilon_r: np.ndarray = np.vstack([self.delta * np.cos(phi), np.sin(phi), np.zeros_like(r)]).T

            # This vector should be scaled by r, but because we are normalising it later, we avoid the multiplication by r.
            epsilon_phi: np.ndarray = np.vstack([-self.delta * np.sin(phi), np.cos(phi), np.zeros_like(r)]).T
            epsilon_z: np.ndarray = np.vstack([np.zeros_like(r), np.zeros_like(r), np.ones_like(r)]).T

            # Normalise the vectors.
            epsilon_r /= np.linalg.norm(epsilon_r, axis=1, keepdims=True)
            epsilon_phi /= np.linalg.norm(epsilon_phi, axis=1, keepdims=True)

            basis = np.stack([epsilon_r, epsilon_phi, epsilon_z], axis=-1)
            return np.matmul(self.psi_rotation_matrix, basis)

    def convert_elliptical_to_cartesian_vector(self, v_r: float, v_phi: float, v_z: float, r: float, phi: float) -> np.ndarray:
        # Because the change of basis is not constant, we have to get the basis change matrix at this (r, phi) coordinate.
        elliptical_to_cartesian_basis_change_matrix: np.ndarray = self.get_elliptical_unit_basis(r=r, phi=phi)
        if isinstance(v_r, float) and isinstance(v_phi, float) and isinstance(v_z, float):
            return np.array([v_r, v_phi, v_z]) @ elliptical_to_cartesian_basis_change_matrix.T
        else:
            vector_array = np.array([v_r, v_phi, v_z]).T
            return np.einsum('ijk,ik->ij', elliptical_to_cartesian_basis_change_matrix, vector_array)

    def get_elliptical_basis_metric(self, r: float, phi: float) -> None:
        elliptical_basis = self.get_elliptical_unit_basis(r, phi)
        elliptical_basis_metric = elliptical_basis.T @ elliptical_basis
        return elliptical_basis_metric

    def evaluate_ellipse_equation(self, x: float, y: float, z: float) -> float:
        return (self.delta * x) ** 2 + y**2 - self.R**2

    def convert_elliptical_to_cartesian_cordinates(
        self, r: float | np.ndarray, phi: float | np.ndarray, z: float | np.ndarray
    ) -> np.ndarray:
        """Convert from elliptical coordinates (r, phi, z) to Cartesian."""
        x = self.delta * r * np.cos(phi)
        y = r * np.sin(phi)
        return (self.psi_rotation_matrix @ np.array([x, y, z])).T

    def convert_cartesian_to_elliptical_coordinates(self, x: float | np.ndarray, y: float | np.ndarray, z: float | np.ndarray) -> np.ndarray:
        """Convert Cartesian coordinates (x, y, z) to elliptical coordinates (r, phi, z), using the delta parameter.
        Note that when delta = 1, this is the normal Cartesian to cylindrical / polar coordinate change.

        Args:
            x (float | np.ndarray): x-coordinate.
            y (float | np.ndarray): y-coordinate.
            z (float | np.ndarray): z-coordinate.

        Returns:
            np.ndarray: array containing the corresponding elliptical coordinates [r, phi, z].
        """

        # Start by computing the angle phi
        phi = np.atan2(y, self.delta * x) - self.psi

        # Pre-compute its cosine and sine for performance and readibility purposes.
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # Compute the radial coordinate.
        r = np.sqrt((x * x + y * y) / (self.delta_squared * cos_phi * cos_phi + sin_phi * sin_phi))

        # The z coordinate remains unchanged.
        elliptical_coordinates = np.stack([r, phi, z]).T

        # zero_tolerance: float = 1e-3
        # zero_x_and_y_mask = np.logical_and(np.abs(x) < zero_tolerance, np.abs(y) < zero_tolerance)
        # cartesian_coordinates[zero_x_and_y_mask, :] = np.array([0, 0, z])

        return elliptical_coordinates

    def compute_scale_factors(self, r: float, phi: float) -> np.ndarray:
        """Compute the scale factors h_r, h_phi, h_z."""
        elliptical_basis: np.ndarray = self.get_elliptical_unit_basis(r=r, phi=phi + self.psi)
        return np.sqrt(np.sum(np.square(elliptical_basis), axis=0))
    
    def get_h(self, phi: float) -> float:
        """Calculate the "h" factor, defined so that (h_phi)^2 = r^2 * h^2 (eq. 5 in the paper).
        Note that h is not constant, but a function of phi. Also note that h is never zero."""
        sin_phi: float = math.sin(phi)
        cos_phi: float = math.cos(phi)
        return math.sqrt(self.delta_squared * sin_phi * sin_phi + cos_phi * cos_phi)

    def get_chi(self, phi: float | None = None, h: float | None = None) -> float:
        if phi is None and h is None:
            raise ValueError("Parameters 'phi' and 'h' cannot be both None.")
        
        if h is None:
            h: float = self.get_h(phi)

        return (self.delta_squared + 1) / (h * h)
    
    def get_magnitude_elliptical_basis(self,
                                       v_r: float | np.ndarray,
                                       v_phi: float | np.ndarray, 
                                       v_z: float | np.ndarray,
                                       r: float | np.ndarray,
                                       phi: float | np.ndarray) -> float | np.ndarray:
        cartesian_vector = self.convert_elliptical_to_cartesian_vector(v_r, v_phi, v_z, r, phi)
        return self.cartesian_vector_magnitude(cartesian_vector[:, 0], cartesian_vector[:, 1], cartesian_vector[:, 2])

    def radial_coordinate_sweep(self, phi: float = 0,
                                num_points: int = 51,
                                normalise_radial_coordinate: bool = False,
                                two_fold: bool = False,
                                plot: bool = False,
                                fig_size: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_range: np.ndarray = np.linspace(0, self.R, num_points, endpoint=True)
        phi_range: np.ndarray = phi * np.ones_like(r_range)
        B_field: np.ndarray = np.zeros((len(r_range), 3))
        J_field: np.ndarray = np.zeros((len(r_range), 3))

        for idx, r in enumerate(r_range):
            B_field[idx, :] = self.get_magnetic_field_elliptical_coordinates(r=r, phi=phi)
            J_field[idx, :] = self.get_current_density_field_elliptical_coordinates(r=r, phi=phi)
        
        F_field = self.get_force_density_field_elliptical_coordinates(J_field=J_field, B_field=B_field)

        if two_fold:
            r_range = np.concatenate([-np.flip(r_range[1:]), r_range])
            phi_range = np.concatenate([-np.flip(phi_range[1:]), phi_range])
            B_field = np.vstack([np.flipud(B_field[1:, :]), B_field])
            J_field = np.vstack([np.flipud(J_field[1:, :]), J_field])
            F_field = np.vstack([np.flipud(F_field[1:, :]), F_field])

        # Extract the poloidal and axial components of the magnetic field and current density, and calculate the magnitude of the fields.
        B_poloidal: np.ndarray = B_field[:, 1]
        B_z: np.ndarray = B_field[:, 2]
        B_magnitude: np.ndarray = self.get_magnitude_elliptical_basis(B_field[:, 0], B_field[:, 1], B_field[:, 2], r_range, phi_range)

        J_poloidal: np.ndarray = J_field[:, 1]
        J_z: np.ndarray = J_field[:, 2]
        J_magnitude: np.ndarray = np.sqrt(np.square(J_poloidal) + np.square(J_z))

        F_radial: np.ndarray = F_field[:, 0]
        F_poloidal: np.ndarray = F_field[:, 1]
        F_z: np.ndarray = F_field[:, 2]
        F_magnitude: np.ndarray = np.sqrt(np.square(J_poloidal) + np.square(J_z))

        # Normalise the MFR dimensions by the R parameter.
        if normalise_radial_coordinate:
            r_range /= self.R
            radial_label = "r/R"
        else:
            radial_label = "r [AU]"

        if plot:
            fig, ax = plt.subplots(3, 1, tight_layout=True, figsize=fig_size)
            ax[0].plot(r_range, B_poloidal)
            ax[0].plot(r_range, B_z)
            ax[0].plot(r_range, B_magnitude)
            ax[0].legend(["$B_{\\phi}$", "$B_z$", "$|B|$"])
            ax[0].set_xlim(r_range.min(), r_range.max())
            ax[0].grid(alpha=0.35)
            ax[0].set_title("Radial coordinate sweep")
            ax[0].set_xlabel(radial_label)
            ax[0].set_ylabel("Magnetic field")

            ax[1].plot(r_range, J_poloidal)
            ax[1].plot(r_range, J_z)
            ax[1].plot(r_range, J_magnitude)
            ax[1].legend(["$J_{\\phi}$", "$J_z$", "$|J|$"])
            ax[1].set_xlim(r_range.min(), r_range.max())
            ax[1].grid(alpha=0.35)
            ax[1].set_xlabel(radial_label)
            ax[1].set_ylabel("Current density")

            ax[2].plot(r_range, F_radial)
            ax[2].plot(r_range, F_poloidal)
            ax[2].plot(r_range, F_z)
            ax[2].plot(r_range, F_magnitude)
            ax[2].legend(["$F_r$", "$F_{\\phi}$", "$F_z$", "$|F|$"])
            ax[2].set_xlim(r_range.min(), r_range.max())
            ax[2].grid(alpha=0.35)
            ax[2].set_xlabel(radial_label)
            ax[2].set_ylabel("Force density")

            plt.show()

        return r_range, B_field, J_field
    
    def radial_and_angular_sweep(self,
                                 r_num_points: int = 51,
                                 phi_num_points: int = 51,
                                 normalise_radial_coordinate: bool = False,
                                 plot: bool = False,
                                 fig_size: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Make a radial (r) and angular (phi) sweep of the magnetic field (B) and current density (J).

        Args:
            r_num_points (int, optional): Number of points of the radial sweep. Defaults to 51.
            phi_num_points (int, optional): Number of points of the angular sweep.. Defaults to 51.
            normalise_radial_coordinate (bool, optional): If enabled, normalise the radial coordinate from [0, R] to [0, 1]. Defaults to False.
            plot (bool, optional): If enables, displays an interactive 3D plot with the sweep. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: values for the sweep of r, phi, B and J.
        """
        # Sample the inside of the magnetic flux rope, in uniform steps in r and phi.
        r_range, phi_range = self.get_ellipse_filling(r_num_points, phi_num_points)

        # Initialise variables
        x = np.zeros((r_num_points, phi_num_points))
        y = np.zeros((r_num_points, phi_num_points))
        B_field = np.zeros((r_num_points, phi_num_points, 3))
        J_field = np.zeros((r_num_points, phi_num_points, 3))

        # Loop over the 2D sweep.
        for r_idx, r in enumerate(r_range):
            for phi_idx, phi in enumerate(phi_range):
                B_field[r_idx, phi_idx, :] = self.get_magnetic_field_elliptical_coordinates(r, phi)
                J_field[r_idx, phi_idx, :] = self.get_current_density_field_elliptical_coordinates(r, phi)

                # Convert the elliptical coordinates to Cartesian.
                cartesian_coordinates: np.ndarray = self.convert_elliptical_to_cartesian_cordinates(r, phi, 0)
                x[r_idx, phi_idx] = cartesian_coordinates[0]
                y[r_idx, phi_idx] = cartesian_coordinates[1]

        # Extract the poloidal and axial fields. The radial field is always zero.
        B_phi = B_field[:, :, 1]
        B_z = B_field[:, :, 2]
        J_phi = J_field[:, :, 1]
        J_z = J_field[:, :, 2]

        # Normalise the MFR dimensions by the R parameter.
        if normalise_radial_coordinate:
            x /= self.R
            y /= self.R
            x_label = "x/R"
            y_label = "y/R"
        else:
            x_label = "x [AU]"
            y_label = "y [AU]"

        if plot:
            # Make the 3D plot of the poloidal and axial magnetic field.
            fig, ax = plt.subplots(2, 2, figsize=fig_size, subplot_kw={"projection": "3d"})

            B_phi_abs = np.abs(B_phi)
            norm_1 = matplotlib.colors.Normalize(vmin=np.min(B_phi_abs), vmax=np.max(B_phi_abs))
            scalar_mappable_1 = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm_1)
            ax[0][0].plot_surface(x, y, B_phi, linewidth=0, facecolors=scalar_mappable_1.to_rgba(B_phi_abs), shade=False)
            ax[0][0].set_box_aspect([self.delta, 1, 1])
            ax[0][0].set_title("$B_{\\phi}$")
            ax[0][0].set_xlabel(x_label)
            ax[0][0].set_ylabel(y_label)

            B_z_abs = np.abs(B_z)
            norm_2 = matplotlib.colors.Normalize(vmin=np.min(B_z_abs), vmax=np.max(B_z_abs))
            scalar_mappable_2 = cm.ScalarMappable(cmap=cm.coolwarm, norm=norm_2)
            ax[0][1].plot_surface(x, y, B_z, linewidth=0, facecolors=scalar_mappable_2.to_rgba(B_z), shade=False)
            ax[0][1].set_box_aspect([self.delta, 1, 1])
            ax[0][1].set_title("$B_z$")
            ax[0][1].set_xlabel(x_label)
            ax[0][1].set_ylabel(y_label)

            ax[1][0].plot_surface(x, y, J_phi, cmap=cm.coolwarm)
            ax[1][0].set_box_aspect([self.delta, 1, 1])
            ax[1][0].set_title("$J_{\\phi}$")
            ax[1][0].set_xlabel(x_label)
            ax[1][0].set_ylabel(y_label)

            ax[1][1].plot_surface(x, y, J_z, cmap=cm.coolwarm)
            ax[1][1].set_box_aspect([self.delta, 1, 1])
            ax[1][1].set_title("$J_z$")
            ax[1][1].set_xlabel(x_label)
            ax[1][1].set_ylabel(y_label)

            # Link the axes movement together.
            on_move: callable = self._link_3D_axes_view(fig, ax)
            fig.canvas.mpl_connect(s="motion_notify_event", func=on_move)

            fig.suptitle("Radial and angular sweep")
            plt.show()

        return x, y, B_field, J_field
    
    def get_boundary(self, num_points: int = 101) -> np.ndarray:
        """Calculate the boundary of the ellipse in Cartesian coordinates."""
        phi_range: np.ndarray = np.linspace(start=0, stop=2 * math.pi, num=num_points + 1, endpoint=True)
        r: np.ndarray = self.R * np.ones(num_points + 1)
        z: np.ndarray = np.zeros(num_points + 1)

        return self.convert_elliptical_to_cartesian_cordinates(r=r, phi=phi_range, z=z)

    def get_ellipse_filling(self, r_num_points: int = 51, phi_num_points: int = 51) -> tuple[np.ndarray, np.ndarray]:
        # Create the radial range, without including r = 0.
        r_range: np.ndarray = np.linspace(start=0, stop=self.R, num=r_num_points + 1, endpoint=True)
        r_range = r_range[1:]

        # Create the angular range.
        phi_range: np.ndarray = np.linspace(start=0, stop=2 * math.pi, num=phi_num_points, endpoint=True)

        return r_range, phi_range

    def plot_boundary(
        self, boundary: np.ndarray | None = None,
        vector_dict: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
        normalise_radial_coordinate: bool = False,
        fig_size: tuple[float, float] | None = None,
        axis = None
    ) -> None:
        if boundary is None:
            boundary = self.get_boundary()
        
        if vector_dict is None:
            vector_dict = dict()

        scale_factor: float = 1 / self.R if normalise_radial_coordinate else 1.0

        if axis is None:
            fig, ax = plt.subplots(tight_layout=True, figsize=fig_size)
        else:
            ax = axis
        ax.plot(boundary[:, 0] * scale_factor, boundary[:, 1] * scale_factor)

        for vector_origin, vector_end in vector_dict.values():
            ax.quiver(vector_origin[0] * scale_factor, vector_origin[1] * scale_factor, vector_end[0] * scale_factor, vector_end[1] * scale_factor)

        legend: list[str] = ["MFR boundary"]
        if len(vector_dict) > 0:
            legend.extend(list(vector_dict.keys()))

        ax.legend(legend)
        ax.axhline(y=0, color="k", alpha=0.35)
        ax.axvline(x=0, color="k", alpha=0.35)
        if abs(self.psi) >1e-9:
            # If psi is not zero, the natural axis of the MFR are angled with an angle psi.
            ax.axline((0, 0), slope=math.tan(self.psi), color="g", alpha=0.5)
            ax.axline((0, 0), slope=-1 / math.tan(self.psi), color="g", alpha=0.5)
        ax.set_aspect("equal")
        ax.grid(alpha=0.35)

        if normalise_radial_coordinate:
            ax.set_xlabel("x/R")
            ax.set_ylabel("y/R")
        else:
            ax.set_xlabel("x [AU]")
            ax.set_ylabel("y [AU]")
        
        plt.title("Magnetic flux rope boundary")
        if axis is None:
            plt.show()

    def find_intersection_points_rot(self, y_0: float, theta: float) -> tuple[np.ndarray | None, np.ndarray | None]:
        # The trajectory will pass through [0, y_0, 0].
        # Compute the intersection of the elliptical cylinder with the line of equation (x, y, z) = (0, y_0, 0) + (cos(\theta), 0, \sin(theta))*\lambda
        cos_psi = math.cos(self.psi)
        cos_psi_2 = cos_psi * cos_psi
        sin_psi = math.sin(self.psi)
        sin_psi_2 = sin_psi * sin_psi
        R_2 = self.R * self.R
        delta_2 = self.delta * self.delta

        A = (cos_psi_2 / (delta_2 * R_2)) + (sin_psi_2 / R_2)
        B = -cos_psi * sin_psi * (1 - 1/delta_2) / R_2
        C = (sin_psi_2 / (delta_2 * R_2)) + (cos_psi_2 / R_2)

        # Quadratic equation.
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        y_0_R = y_0 * self.R
        a = A * cos_theta * cos_theta
        b = 2 * B * y_0_R * cos_theta
        c = C * y_0_R * y_0_R - 1

        # Solve the quadratic equation a*lambda^2 + b*lambda + c = 0
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            # No intersection points.
            return None, None
        discriminant_sqrt = math.sqrt(discriminant)
        lambda1 = (-b - discriminant_sqrt) / (2 * a)
        lambda2 = (-b + discriminant_sqrt) / (2 * a)

        # Calculate the intersection points
        intersection1 = np.array([cos_theta * lambda1, y_0_R, sin_theta * lambda1])
        intersection2 = np.array([cos_theta * lambda2, y_0_R, sin_theta * lambda2])

        return intersection1, intersection2

    def _resolve_trajectory(self, v_sc: float, y_0: float, theta: float, time_stencil: int | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """Resolve the entry and exit points of the S/C in the magnetic flux rope.

        Args:
            v_sc (float): Speed of the spacecraft in km/s.
            y_0 (float): Impact parameter
            time_stencil (int | np.ndarray): An integer number of points to sample the trajectory uniformly, or an array of increasing times, not necessarily equispaced.

        Raises:
            ValueError: _description_

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]: _description_
        """
        # Start by finding the rope limits. We solve analitically for the intersection of the trajectory
        # with the elliptical cylinder.

        # The flux rope is left so it has its axis in the z direction.
        # The trajectory of the spacecraft is r(lambda): (x, y, z) = (0, y_0, 0) + (cos(theta), 0, sin(theta))*lambda
        # Where lambda is the parameter. Actually, lambda is just v_sc * (t - t*), where t* is the time at the centre
        # (y = y_0) of the flux rope.

        # Equation of the rope boundary (r = R) when the psi angle is 0, is:
        # x = delta * R * cos(phi)
        # y = R * sin(phi)

        # Which has the equation (x / R)^2 + (y / (delta*R))^2 = R^2.

        # If a rotation is applied, then.


        # If we apply the psi rotation matrix, we get:
        # x' = delta * R * cos(phi) * cos (psi) - R * sin(phi) * sin(psi)
        # y' = delta * R * cos(phi) * sin (psi) + R * sin(phi) * cos(psi)

        # We set y' = y_0 and solve for psi. We obtain two solutions:
        # psi = arctan(cotan(psi) / delta) +- arccos(y_0 / (R * sqrt(delta^2 * sin^2(psi) + cos^2(psi))))
        # We note that the last term is exactly equal to the scale factor h(phi), which can simplify the expression:
        # psi = arctan(cotan(psi) / delta) +- arccos(y_0 / (R * h(psi)))

        # We note that for the circular-cylindrical case (delta = 1), we get a solution that suits what we would expect:
        # psi = arctan(cotan(0)) +- arccos(y_0 / R) = (pi / 2) +- arccos(y_0 / R)

        entry_point, exit_point = self.find_intersection_points_rot(y_0, theta)

        # Start by checking whether the S/C crosses the flux rope.
        if entry_point is None or exit_point is None:
            return [], [], [], [], False

        if isinstance(time_stencil, int):
            # We were supplied with the number of points to sample the trajectory.
            distance_in_flux_rope = np.linalg.norm(np.array(exit_point) - np.array(entry_point))
            time_in_flux_rope: float = (distance_in_flux_rope * self.AU_to_m) / v_sc
            time_range: np.ndarray = np.linspace(start=0, stop=time_in_flux_rope, num=time_stencil, endpoint=True)
        else:
            # We were supplied with an array of increasing times.
            time_range = time_stencil - time_stencil[0]
            
        num_points: int = len(time_range)

        # Build the trajectory inside of the MFR.
        x_tajectory: np.ndarray = entry_point[0] + (exit_point[0] - entry_point[0]) * (time_range / time_range[-1])
        y_trajectory: np.ndarray = y_0 * self.R * np.ones((num_points))
        z_trajectory: np.ndarray = entry_point[2] + (exit_point[2] - entry_point[2]) * (time_range / time_range[-1])

        return time_range, x_tajectory, y_trajectory, z_trajectory, True

    def _validate_crossing_parameters(self, y_0: float, theta: float, time_stencil: int) -> None:
        # Parameter: y_0.
        if not isinstance(y_0, (int, float)):
            raise TypeError("Parameter: y_0 must be an integer or float.")

        if not (-1.0 < y_0 < 1.0):
            raise ValueError("Parameter: y_0 must be in (-1.0, 1.0).")
        
        # Parameter: theta.
        if not isinstance(theta, (int, float)):
            raise TypeError("Parameter: theta must be an integer or float.")
        
        if not (-math.pi/2 < theta < math.pi / 2):
            raise TypeError("Parameter: theta must be in range (-pi/2, pi/2).")
        
        # Parameter: time_stencil.
        if not isinstance(time_stencil, (int, list, tuple, pd.Series, np.ndarray)):
            raise TypeError("Parameter: time_stencil must be an integer or array.")
        
        # Assert number grater that 5 or array increasing.

    def simulate_crossing(self,
                          v_sc: float,
                          y_0: float = 0.0,
                          theta: float = 0.0,
                          time_stencil: int = 51,
                          noise_type: str | None = None,
                          epsilon: float = 0.05,
                          initial_time: float = 0.0,
                          initial_datetime: datetime.datetime | None = None,
                          random_seed: int = 0) -> pd.DataFrame | None:
        """Simulate the crossing of a spacecraft (S/C) through the magnetic flux rope. Simulate the measurements of the S/C through it.
        Currently, only straight trajectories at constant speed are supported.

        Args:
            v_sc (float): Spacecraft speed in km/s.
            y_0 (float, optional): Spacecraft distance from the central axis of the flux rope as a fraction of the flux rope radius. Valid range (-1.0, 1.0). Defaults to 0.0.
            theta (float, optional): Angle between the velocity and the flux rope axis.
            time_stencil (int, optional): Number of data points conforming the time series. Must be >= 5. Defaults to 51.
            noise_type (str | None, optional): _description_. Defaults to None.
            epsilon (float, optional): _description_. Defaults to 0.05.

        Returns:
            pd.DataFrame: Containing the simulated time series data.
        """

        # TODO: Would it be interesting to set the measurement frequency instead of the total number of points?

        # Start by validating the crossing parameters.
        self._validate_crossing_parameters(y_0, theta, time_stencil)

        # Convert the speed from km/s to m/s.
        v_sc_metres_per_second: float = v_sc * 1_000

        # velocity_sc: np.ndarray = v_sc_metres_per_second * np.array([math.cos(theta), 0, math.sin(theta)])

        # Start by resolving the geometrical trajectory of the spacecraft.
        time_range, x_tajectory, y_trajectory, z_trajectory, does_intersect = self._resolve_trajectory(v_sc_metres_per_second, y_0, theta, time_stencil)

        if not does_intersect:
            # The crossing set by the crossing parameters does not intersect the flux-rope.
            return None

        num_points: int = len(time_range)

        # Initialise the magnetic and current density field arrays.
        B_field = np.zeros((num_points, 3))
        J_field = np.zeros((num_points, 3))

        # Iterate over the discretised trajectory.
        for idx in range(num_points):
            x = x_tajectory[idx]
            y = y_trajectory[idx]
            z = z_trajectory[idx]

            elliptical_coordinates: np.ndarray = self.convert_cartesian_to_elliptical_coordinates(x, y, z)
            r: float = elliptical_coordinates[0]
            phi: float = elliptical_coordinates[1]
            z: float = elliptical_coordinates[2]

            B_field_elliptical: np.ndarray = self.get_magnetic_field_elliptical_coordinates(r, phi)
            J_field_elliptical: np.ndarray = self.get_current_density_field_elliptical_coordinates(r, phi)

            # Convert the elliptical components of the vector fields back to Cartesian.
            B_field[idx, :] = self.convert_elliptical_to_cartesian_vector(
                B_field_elliptical[0], B_field_elliptical[1], B_field_elliptical[2], r=r, phi=phi
            )
            J_field[idx, :] = self.convert_elliptical_to_cartesian_vector(
                J_field_elliptical[0], J_field_elliptical[1], J_field_elliptical[2], r=r, phi=phi
            )

        close_to_zero_tolerance: float = 1e-15
        B_field[np.abs(B_field) < close_to_zero_tolerance] = 0
        J_field[np.abs(J_field) < 1e-6] = 0

        # Add noise to simulate measurement error, if the user wants it.
        if noise_type is not None:
            noise_generator: RandomNoise = self.get_noise_generator(noise_type, epsilon, random_seed)

            B_field[:, 0] += noise_generator.generate_noise(num_points)
            B_field[:, 1] += noise_generator.generate_noise(num_points)
            B_field[:, 2] += noise_generator.generate_noise(num_points)
            J_field[:, 0] += noise_generator.generate_noise(num_points)
            J_field[:, 1] += noise_generator.generate_noise(num_points)
            J_field[:, 2] += noise_generator.generate_noise(num_points)

        # We need to change the magnetic field and current density to the new coordinates (rotated by theta).
        # We can do this by applying the rotation matrix.
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        rotation_matrix = np.array([[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]])

        B_field = B_field @ rotation_matrix
        J_field = J_field @ rotation_matrix

        # Join all the simulated measurements on a pandas data frame.
        df = pd.DataFrame()
        df["time"] = initial_time + time_range
        if initial_datetime is not None:
            df["datetime"] = initial_datetime + pd.to_timedelta(time_range, unit="s")

        df["x"] = x_tajectory
        df["y"] = y_trajectory
        df["z"] = z_trajectory

        df["B_x"] = B_field[:, 0]
        df["B_y"] = B_field[:, 1]
        df["B_z"] = B_field[:, 2]
        df["B"] = self.cartesian_vector_magnitude(B_field[:, 0], B_field[:, 1], B_field[:, 2])

        df["J_x"] = J_field[:, 0]
        df["J_y"] = J_field[:, 1]
        df["J_z"] = J_field[:, 2]
        df["J"] = self.cartesian_vector_magnitude(J_field[:, 0], J_field[:, 1], J_field[:, 2])

        F_field = self.get_force_density_field_elliptical_coordinates(B_field=B_field, J_field=J_field)
        df["F_x"] = F_field[:, 0]
        df["F_y"] = F_field[:, 1]
        df["F_z"] = F_field[:, 2]
        df["F"] = self.cartesian_vector_magnitude(F_field[:, 0], F_field[:, 1], F_field[:, 2])

        return df

    def plot_vs_time(self,
                     data: pd.DataFrame,
                     magnitude_names: str | list[str],
                     colour: str | list[str],
                     time_units: str = "s",
                     datetime_axis: bool = False,
                     marker: str = "o",
                     linestyle: str = "-",
                     markersize: str = 4,
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

        time_min = np.min(time)
        time_max = np.max(time)

        if not axis_provided:
            _, ax = plt.subplots(1, 1, tight_layout=True)

        for idx, magnitude_name in enumerate(magnitude_names):
            # Extract the magnitude to plot from the dataframe, and its units.
            magnitude_to_plot = data[magnitude_name].to_numpy(copy=True)
            magnitude_units = self._units[magnitude_name]
            ax.plot(time, magnitude_to_plot, marker=marker, linestyle=linestyle, markersize=markersize, color=colour[idx])
            if datetime_axis:
                ax.set_xlabel("datetime")
            else:
                ax.set_xlabel(f"time [{time_units}]")
            ax.set_ylabel(f"${magnitude_name}$ [{magnitude_units}]")
            ax.grid(alpha=0.35)
            ax.set_xlim(time_min, time_max)
            # plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)

        plt.legend([f"${mag}$" for mag in magnitude_names])

        if not axis_provided:
            plt.show()
        else:
            return ax
        
    @abc.abstractmethod
    def get_magnetic_field_elliptical_coordinates(self, r: float, phi: float) -> np.ndarray:
        """Implement the magnetic field components of the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_current_density_field_elliptical_coordinates(self, r: float, phi: float) -> np.ndarray:
        """Implement the current density components of the model."""
        raise NotImplementedError

    def get_force_density_field_elliptical_coordinates(self, r: float | None = None, phi: float | None = None, J_field: np.ndarray | None = None, B_field: np.ndarray | None = None) -> np.ndarray:
        """If the subclass does not implement the force explicitly, fall into this function and calculate it from the magnetic and
        current density fields."""
        # Check if we have been passed the pre-computed J and B fields. This helps with performance.
        if J_field is None:
            J_field = self.get_current_density_field_elliptical_coordinates(r, phi)
        
        if B_field is None:
            B_field = self.get_magnetic_field_elliptical_coordinates(r, phi)
        
        return self.get_force_density_field_from_unit_right_handed_orthogonal_basis(J_field, B_field)
