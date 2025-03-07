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


def test_quadrants() -> None:
    for R in np.linspace(0.1, 1.0, 10, endpoint=True):
        for psi in np.linspace(0, math.pi, 10, endpoint=False):
            for delta in np.linspace(0.1, 1.0, 10, endpoint=True):
                model = ECModel(delta=delta, psi=psi, R=R)
                x = R*delta*math.cos(psi)
                y = R*delta*math.sin(psi)
                z = 0.0

                ellip = model.convert_cartesian_to_elliptical_coordinates(x, y, z)
                r = ellip[0]
                phi = ellip[1]
                z_ellip = ellip[2]

                assert np.isclose(R, r, atol=1e-8)
                assert np.isclose(0, phi, atol=1e-8)
                assert np.isclose(0, z_ellip, atol=1e-8)


def test_z_unchanged() -> None:
    pass

def test_elliptical_to_cartesian() -> None:
    R = 0.1
    psi = math.pi/10
    delta = 0.9

    # Create an instance of the model with the given delta, psi and R.
    model = ECModel(delta=delta, psi=psi, R=R)

    p_elliptical = np.array([R, 0, 0])
    p_cartesian = model.convert_elliptical_to_cartesian_cordinates(p_elliptical[0], p_elliptical[1], p_elliptical[2])

    assert np.allclose(p_cartesian, np.array([R*delta*math.cos(psi), R*delta*math.sin(psi), 0]), atol=1e-8)

