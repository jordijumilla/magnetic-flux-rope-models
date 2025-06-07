import pytest
import itertools
import numpy as np
import math
from magnetic_flux_rope_models.EllipticalCylindricalModel import EllipticalCylindricalModel


# Define parameter grids
R_range = np.linspace(0.1, 1.0, 10, endpoint=True)
delta_range = np.linspace(0.1, 1.0, 10, endpoint=True)
psi_range = np.deg2rad([0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5])

# Generate all combinations
all_param_combinations: list = list(itertools.product(R_range, delta_range, psi_range))

# Define the test tolerance
test_tolerance: float = 1e-8

@pytest.mark.parametrize("R, delta, psi", all_param_combinations)
def test_identity_change_of_coordinates_cartesian_to_elliptical(R: float, delta: float, psi: float) -> None:
    # Create an instance of the model with the given R, delta and psi.
    model = EllipticalCylindricalModel(R=R, delta=delta, psi=psi)
    
    # Convert the cartesian coordinates (0, R, 0) to elliptical coordinates.
    x = 0.0
    y = model.R
    z = 0.0
    elliptical_coordinates = model.convert_cartesian_to_elliptical_coordinates(x, y, z)

    # Convert the elliptical coordinates back to Cartesian coordinates.
    cartesian_coordinates = model.convert_elliptical_to_cartesian_coordinates(elliptical_coordinates[0], elliptical_coordinates[1], elliptical_coordinates[2])

    # Check that we recover the original Cartesian coordinates.
    assert np.allclose([x, y, z], cartesian_coordinates, atol=test_tolerance)

@pytest.mark.parametrize("R, delta, psi", all_param_combinations)
def test_identity_change_of_coordinates_elliptical_to_cartesian(R: float, delta: float, psi: float) -> None:
    # Create an instance of the model with the given R, delta and psi.
    model = EllipticalCylindricalModel(R=R, delta=delta, psi=psi)

    # Convert the elliptical coordinates (R, 0, 0) to cartesian coordinates.
    cartesian_coordinates = model.convert_elliptical_to_cartesian_coordinates(r=model.R, phi=0.0, z=0.0)

    # Convert the Cartesian coordinates back to elliptical coordinates.
    elliptical_coordinates = model.convert_cartesian_to_elliptical_coordinates(cartesian_coordinates[0], cartesian_coordinates[1], cartesian_coordinates[2])

    # Check that we recover the original elliptical coordinates.
    assert np.allclose([model.R, 0, 0], elliptical_coordinates, atol=test_tolerance)

def get_quadrant_point(R: float, delta: float, psi: float, quadrant: int) -> tuple[float, float]:
    """Get a point in the specified quadrant of the elliptical elliptical-cylindrical coordinates."""
    if quadrant == 1:
        x = R*delta*math.cos(psi)
        y = R*delta*math.sin(psi)
    elif quadrant == 2:
        x = -R*math.sin(psi)
        y = R*math.cos(psi)
    elif quadrant == 3:
        x = -R*delta*math.cos(psi)
        y = -R*delta*math.sin(psi)
    elif quadrant == 4:
        x = R*math.sin(psi)
        y = -R*math.cos(psi)
    else:
        raise ValueError(f"Invalid quadrant: {quadrant}")

    return x, y

@pytest.mark.parametrize("R, delta, psi", all_param_combinations)
def test_quadrants(R: float, delta: float, psi: float) -> None:
    # Create an instance of the model with the given delta, psi and R.
    model = EllipticalCylindricalModel(delta=delta, psi=psi, R=R)

    # Test each quadrant.
    for quadrant, expected_phi in zip([1, 2, 3, 4], [0, math.pi/2, math.pi, 3*math.pi/2]):
        x, y = get_quadrant_point(R=R, delta=delta, psi=psi, quadrant=quadrant)

        ellip = model.convert_cartesian_to_elliptical_coordinates(x=x, y=y, z=0.0)
        r = ellip[0]
        phi = ellip[1]
        z_ellip = ellip[2]

        # The radial coordinate should be equal to R.
        assert np.isclose(R, r, atol=test_tolerance)

        # The azimuthal coordinate should be equal to the expected value.
        # For the third quadrant, the azimuthal coordinate can be either 180 degrees or -180 degrees.
        if quadrant == 3:
            assert np.isclose(expected_phi, phi, atol=test_tolerance) or np.isclose(-expected_phi, phi, atol=test_tolerance)
        else:
            assert np.isclose(expected_phi, phi, atol=test_tolerance)
        assert np.isclose(0, z_ellip, atol=test_tolerance)

@pytest.mark.parametrize("R, delta, psi", all_param_combinations)
def test_elliptical_to_cartesian(R: float, delta: float, psi: float) -> None:
    # Create an instance of the model with the given delta, psi and R.
    model = EllipticalCylindricalModel(delta=delta, psi=psi, R=R)

    # The z-value should not change.
    for z in [-1, 0, 1]:
        for quadrant, phi in zip([1, 2, 3, 4], [0, math.pi/2, math.pi, 3*math.pi/2]):
            # Convert the elliptical coordinates to cartesian coordinates.
            p_cartesian = model.convert_elliptical_to_cartesian_coordinates(r=R, phi=phi, z=z)
            
            # The expected cartesian coordinates for the point (R, 0, 0) in elliptical coordinates.
            expected_coordinates = get_quadrant_point(R=R, delta=delta, psi=psi, quadrant=quadrant)
            
            # Check that the cartesian coordinates are close to the expected coordinates.
            assert np.isclose(p_cartesian[0], expected_coordinates[0], atol=test_tolerance)
            assert np.isclose(p_cartesian[1], expected_coordinates[1], atol=test_tolerance)
            assert np.isclose(p_cartesian[2], z, atol=test_tolerance)

@pytest.mark.parametrize("R, delta, psi", all_param_combinations)
def test_elliptical_unit_basis_scalar(R: float, delta: float, psi: float) -> None:
    # Create an instance of the EllipticalCylindricalModel with specific parameters.
    ec_model = EllipticalCylindricalModel(R=R, delta=delta, psi=psi)

    # Get the elliptical basis vectors at r=R and phi=0.
    unit_basis = ec_model.get_elliptical_basis(r=ec_model.R, phi=0.0, unit_basis=True)

    # Check that the basis vectors are the expected theoretical values.
    assert np.allclose(unit_basis[:, 0], np.array([np.cos(ec_model.psi), np.sin(ec_model.psi), 0]), atol=1e-8)
    assert np.allclose(unit_basis[:, 1], np.array([-np.sin(ec_model.psi), np.cos(ec_model.psi), 0]), atol=1e-8)
    assert np.allclose(unit_basis[:, 2], np.array([0, 0, 1]), atol=1e-8)
