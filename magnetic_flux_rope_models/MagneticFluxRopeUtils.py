import numpy as np

def compute_intermediate_variance_axis(B_field: np.ndarray, normalise: bool = False) -> np.ndarray:
    """Compute the intermediate variance axis of a magnetic field vector.
    This function computes the intermediate variance axis of a magnetic field vector
    using the covariance matrix of the magnetic field vector. The intermediate variance axis
    is the eigenvector corresponding to the intermediate eigenvalue of the covariance matrix.
    
    Args:
        B_field (np.ndarray): A 2D array of shape (N, 3) containing the magnetic field vectors.
        normalise (bool): If True, the output vector is normalised to unit length. Default is False.
    
    Returns:
        np.ndarray: A 1D array of shape (3,) containing the intermediate variance axis of the magnetic field vector.
    """
    if normalise:
        B_field_mag = np.linalg.norm(B_field, axis=1)
        B_field = B_field / B_field_mag[:, np.newaxis]

    M = np.cov(B_field.transpose())
    eigenvalues, W = np.linalg.eig(M)

    index_min: int = np.argmin(eigenvalues)
    index_max: int = np.argmax(eigenvalues)
    index_intermediate: int = 3 - index_min - index_max

    estimated_ax = W[:, index_intermediate]

    # Correct the sign.
    if estimated_ax[2] < 0:
        estimated_ax *= (-1)

    return estimated_ax
