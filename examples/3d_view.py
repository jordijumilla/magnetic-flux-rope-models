import sys
sys.path.append("../MagneticFluxRopeModels")

import numpy as np
import matplotlib.pyplot as plt
from MagneticFluxRopeModels.CCModel import CCModel
import math


def draw_sphere(ax, center=(0, 0, 0), radius=1, color='yellow'):
    N = 50
    u = np.linspace(0, 2 * np.pi, N)
    v = np.linspace(0, np.pi, N)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, shade=True, alpha=0.8)

def draw_cylinder(ax, start, end, radius=0.5, color='gray', alpha=1.0):
    v = np.array(end) - np.array(start)
    mag = np.linalg.norm(v)
    v = v / mag

    N = 50
    not_v = np.array([1, 0, 0]) if v[0] == 0 else np.array([0, 1, 0])
    n1 = np.cross(v, not_v)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v, n1)

    t = np.linspace(0, 2 * np.pi, N)
    z = np.linspace(0, mag, N)
    t, z = np.meshgrid(t, z)

    X, Y, Z = [start[i] + v[i] * z + radius * np.cos(t) * n1[i] + radius * np.sin(t) * n2[i] for i in range(3)]
    ax.plot_surface(X, Y, Z, color=color, shade=True, alpha=alpha)

def draw_axes(ax, origin, basis: dict[str, np.ndarray], r: float = 1.0):
    for i, ((label, vec), c) in enumerate(zip(basis.items(), ["r", "g", "b"])):
        ax.quiver(*origin, vec[0], vec[1], vec[2], color=c, label=label)
        ax.text(origin[0] + r*vec[0], origin[1] + r*vec[1], origin[2] + r*vec[2], label, color=c)

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def draw_angle_arc(ax, A, B, C, radius=1.0, n_points=100, color="orange"):
    # Normalize vectors
    v1 = (B - A) / np.linalg.norm(B - A)
    v2 = (C - A) / np.linalg.norm(C - A)

    # Compute rotation axis
    normal = np.cross(v1, v2)
    if np.linalg.norm(normal) < 1e-8:
        return  # vectors are collinear, no visible angle

    normal = normal / np.linalg.norm(normal)

    # Compute angle between vectors
    angle = np.arccos(np.clip(np.dot(v1, v2), -1, 1))

    # Create arc points
    theta = np.linspace(0, angle, n_points)
    arc_points = np.array([np.cos(t) * v1 + np.sin(t) * np.cross(normal, v1) for t in theta])
    arc_points = A + radius * arc_points

    # Plot the arc
    ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], color=color)

    return np.mean(arc_points, axis=0)

def main():
    angle_type = "theta"  # or "theta"
    cc_model = CCModel(R=1)

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Draw global axes
    gse_basis = {r"$x^{GSE}$": np.array([1, 0, 0]), r"$y^{GSE}$": np.array([0, 1, 0]), r"$z^{GSE}$": np.array([0, 0, 1])}
    draw_axes(ax, origin=np.array([0, 0, 0]), basis=gse_basis, r=1.15)

    # Draw the Sun (sphere)
    sphere_centre = [5, 0, 0]
    sphere_radius = 1
    draw_sphere(ax, center=sphere_centre, radius=sphere_radius, color="yellow")
    ax.text(sphere_centre[0], sphere_centre[1], sphere_centre[2] + 1.5*sphere_radius, "Sun", color="black", fontsize=12, va="center", ha="center")

    # Draw the cylinder and its local axes
    local_basis_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

    if angle_type == "gamma":
        gamma = math.radians(-45)
        theta = math.radians(0)
    else:
        gamma = math.radians(0)
        theta = math.radians(45)
    rotation_matrix: np.ndarray = cc_model.convert_local_to_gse_coordinates(gamma, theta)
    local_basis_matrix_trans = rotation_matrix @ local_basis_matrix

    local_basis = {
        r"$x^{local}$": local_basis_matrix_trans[:, 0],
        r"$y^{local}$": local_basis_matrix_trans[:, 2],
        r"$z^{local}$": local_basis_matrix_trans[:, 1]
    }

    cylinder_centre = np.array([2.5, 0, 0])
    draw_axes(ax, origin=cylinder_centre, basis=local_basis, r=1.15)

    cylinder_length = 3.0
    cylinder_start = cylinder_centre + (cylinder_length/2)*local_basis_matrix_trans[:, 1]
    cylinder_end = cylinder_centre - (cylinder_length/2)*local_basis_matrix_trans[:, 1]
    cylinder_radius = 0.4
    draw_cylinder(ax, start=cylinder_start, end=cylinder_end, radius=cylinder_radius, alpha=0.4)

    ax.plot([0, sphere_centre[0] - sphere_radius], [0, 0], [0, 0], color="black", alpha=0.5, linestyle="--")  # Line from Sun to cylinder
    ax.scatter(0, 0, 0, color='black', s=100)  # Center of the Sun

    if angle_type == "gamma":
        mean_angle_point = draw_angle_arc(ax, A=cylinder_centre, B=cylinder_centre + local_basis_matrix_trans[:, 0], C=sphere_centre, radius=0.6, n_points=21, color="black")
        ax.text(1.05*mean_angle_point[0], mean_angle_point[1] - 0.1, mean_angle_point[2], r"$\gamma$", color="k", fontsize=12, va="center", ha="center")
    else:
        mean_angle_point = draw_angle_arc(ax, A=cylinder_centre, B=cylinder_centre + local_basis_matrix_trans[:, 1], C=cylinder_centre + gse_basis[r"$y^{GSE}$"], radius=0.6, n_points=21, color="black")
        ax.text(1.05*mean_angle_point[0], mean_angle_point[1] + 0.1, mean_angle_point[2], r"$\theta$", color="k", fontsize=12, va="center", ha="center")
        ax.plot([cylinder_centre[0], cylinder_centre[0]], [cylinder_centre[1] - cylinder_length/2, cylinder_centre[1] + cylinder_length/2], [0, 0], color="black", alpha=0.5, linestyle="--")  # Line from Sun to cylinder
    set_axes_equal(ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Visualization with Sun, Cylinder, and Local Axes")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # ax.xaxis.pane.set_visible(False)
    # ax.yaxis.pane.set_visible(False)
    # ax.zaxis.pane.set_visible(False)
    # ax.set_axis_off()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()