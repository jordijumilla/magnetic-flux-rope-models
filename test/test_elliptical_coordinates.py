from ECModel import ECModel
import numpy as np
import math


def test_identity() -> None:
    for delta in np.linspace(0.1, 1.0, 10, endpoint=True):
        for psi in np.linspace(0, math.pi, 10, endpoint=False):
            model = ECModel(delta=delta, psi=psi)
            x = 0.0
            y = 0.05
            z = 0.0

            ellip = model.convert_cartesian_to_elliptical_coordinates(x, y, z)
            cart = model.convert_elliptical_to_cartesian_cordinates(ellip[0], ellip[1], ellip[2])

            assert np.allclose([x, y, z], cart, atol=1e-8)


def get_quadrant_point(delta: float, psi: float, R: float, quadrant: int) -> tuple[float, float]:
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
        raise ValueError("Invalid quadrant")

    return x, y


def test_quadrants() -> None:
    for R in np.linspace(0.1, 1.0, 10, endpoint=True):
        for psi in np.linspace(0, math.pi, 10, endpoint=False):
            for delta in np.linspace(0.1, 1.0, 10, endpoint=True):
                model = ECModel(delta=delta, psi=psi, R=R)
                for quadrant, expected_phi in zip([1, 2, 3, 4], [0, math.pi/2, math.pi, -math.pi/2]):
                    x, y = get_quadrant_point(delta, psi, R, quadrant)

                    ellip = model.convert_cartesian_to_elliptical_coordinates(x, y, 0.0)
                    r = ellip[0]
                    phi = ellip[1]
                    z_ellip = ellip[2]

                    assert np.isclose(R, r, atol=1e-8)
                    if quadrant == 3:
                        assert np.isclose(expected_phi, phi, atol=1e-8) or np.isclose(-expected_phi, phi, atol=1e-8)
                    else:
                        assert np.isclose(expected_phi, phi, atol=1e-8)
                    assert np.isclose(0, z_ellip, atol=1e-8)


def test_z_unchanged() -> None:
    pass


def test_elliptical_to_cartesian() -> None:
    for R in np.linspace(0.1, 1.0, 10, endpoint=True):
        for psi in np.linspace(0, math.pi, 10, endpoint=False):
            for delta in np.linspace(0.1, 1.0, 10, endpoint=True):
                # Create an instance of the model with the given delta, psi and R.
                model = ECModel(delta=delta, psi=psi, R=R)

                p_cartesian = model.convert_elliptical_to_cartesian_cordinates(R, 0, 0)
                assert np.allclose(p_cartesian, np.array([R*delta*math.cos(psi), R*delta*math.sin(psi), 0]), atol=1e-8)
