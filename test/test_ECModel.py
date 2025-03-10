from MagneticFluxRopeModels.ECModel import ECModel
import numpy as np

def test_B_z_0():
    for delta in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for psi in np.deg2rad([0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]):
            for B_z_0 in [5.0, 10.0, 15.0]:
                for tau in [0.5, 0.75, 1.0, 1.25, 1.5]:
                    ec_model = ECModel(delta=delta, psi=psi, B_z_0=B_z_0, tau=tau)
                    magnetic_field_centre = ec_model.get_magnetic_field_elliptical_coordinates(0, 0)
                    assert abs(magnetic_field_centre[2] - ec_model.B_z_0) < 1e-8


def test_tau():
    for delta in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for psi in np.deg2rad([0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]):
            for B_z_0 in [5.0, 10.0, 15.0]:
                for tau in [0.5, 0.75, 1.0, 1.25, 1.5]:
                    ec_model = ECModel(delta=delta, psi=psi, B_z_0=B_z_0, tau=tau)
                    magnetic_field_centre = ec_model.get_magnetic_field_elliptical_coordinates(0, 0)
                    magnetic_field_border = ec_model.get_magnetic_field_elliptical_coordinates(ec_model.R, 0)
                    
                    theoretical_tau = magnetic_field_centre[2] / (magnetic_field_centre[2] - magnetic_field_border[2])
                    assert abs(theoretical_tau - ec_model.tau) < 1e-8
