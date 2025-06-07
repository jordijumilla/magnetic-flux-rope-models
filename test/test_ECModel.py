import pytest
import itertools
from magnetic_flux_rope_models.ECModel import ECModel
import numpy as np

# Define parameter grids
delta_range = np.linspace(0.1, 1.0, 10, endpoint=True)
psi_range = np.deg2rad([0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5])
B_z_0_range = [5.0, 10.0, 15.0]
tau_range = [0.5, 0.75, 1.0, 1.25, 1.5]
C_nm = [0.5, 1, 1.5, 2.0]
n_range = [1, 2, 3]
m_range = [0, 1, 2]

# Generate all combinations
all_param_combinations: list = list(itertools.product(delta_range, psi_range, B_z_0_range, tau_range, C_nm, n_range, m_range))

# Define the test tolerance
test_tolerance: float = 1e-8

@pytest.mark.parametrize("delta, psi, B_z_0, tau, C_nm, n, m", all_param_combinations)
def test_B_z_0(delta: float, psi: float, B_z_0: float, tau: float, C_nm: float, n: int, m: int) -> None:
    """
    Test that the B_z_0 parameter is correctly set in the ECModel.
    For a range of delta, psi, B_z_0, and tau values, check that the z-component
    of the magnetic field at the center (0, 0) matches B_z_0 within a tight tolerance.
    """
    ec_model: ECModel = ECModel(delta=delta, psi=psi, B_z_0=B_z_0, tau=tau, C_nm=C_nm, n=n, m=m)

    # Assert that the B_z_0 parameter in the model matches the input B_z_0
    assert ec_model.B_z_0 == B_z_0

    # Get the magnetic field at the center (r = 0, phi = any)
    magnetic_field_centre: np.ndarray = ec_model.get_magnetic_field_elliptical_coordinates(r=0, phi=0)

    # Assert that the z-component at the center equals B_z_0
    assert abs(magnetic_field_centre[2] - ec_model.B_z_0) < test_tolerance

@pytest.mark.parametrize("delta, psi, B_z_0, tau, C_nm, n, m", all_param_combinations)
def test_tau(delta: float, psi: float, B_z_0: float, tau: float, C_nm: float, n: int, m: int) -> None:
    """
    Test that the tau parameter is correctly set in the ECModel.
    For a range of delta, psi, B_z_0, and tau values, check that the theoretical tau,
    computed from the magnetic field at the center and border, matches the model's tau.
    """
    # Create the ECModel with the given parameters
    ec_model: ECModel = ECModel(n=n, m=m, delta=delta, psi=psi, B_z_0=B_z_0, tau=tau, C_nm=C_nm)

    # Assert that the tau parameter in the model matches the input tau
    assert ec_model.tau == tau

    # Get the magnetic field at the center and at the border (r = R, phi = 0)
    magnetic_field_centre: np.ndarray = ec_model.get_magnetic_field_elliptical_coordinates(r=0, phi=0)
    magnetic_field_border: np.ndarray = ec_model.get_magnetic_field_elliptical_coordinates(r=ec_model.R, phi=0)

    # Theoretical tau from the field values
    theoretical_tau: float = magnetic_field_centre[2] / (magnetic_field_centre[2] - magnetic_field_border[2])
    
    # Assert that the theoretical tau is close to the model's tau
    assert abs(theoretical_tau - ec_model.tau) < test_tolerance

@pytest.mark.parametrize("delta, psi, B_z_0, tau, C_nm, n, m", all_param_combinations)
def test_C_nm(delta: float, psi: float, B_z_0: float, tau: float, C_nm: float, n: int, m: int) -> None:
    """
    Test that the C_nm parameter is correctly set in the ECModel.
    For a range of delta, psi, B_z_0, and tau values, check that the C_nm value
    is positive and matches the expected value in the model.
    """
    # Create the ECModel with the given parameters
    ec_model: ECModel = ECModel(n=n, m=m, delta=delta, psi=psi, B_z_0=B_z_0, tau=tau, C_nm=C_nm)
    
    # Assert that the C_nm parameter in the model matches the input C_nm
    assert ec_model.C_nm == C_nm

    # Theoretical C_nm calculation based on the model parameters        
    theoretical_C_nm: float = (ec_model.alpha_n / ec_model.beta_m) * (ec_model.R * ec_model.AU_to_m) ** (ec_model.n - ec_model.m)
    
    # Assert that the theoretical C_nm is positive and matches the model's C_nm
    assert theoretical_C_nm > 0
    assert abs(theoretical_C_nm - ec_model.C_nm) < test_tolerance
