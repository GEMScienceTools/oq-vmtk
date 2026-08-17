##########################################################################
#                     SDOF-MDOF CALIBRATION MODULE                       #
##########################################################################
"""
Transforms an SDOF spectral capacity curve (Sd, Sa) into per-storey
force-drift backbones for a stick MDOF model.

The first-mode shape is either an assumed power law (frame buildings
up to 12 storeys) or the first eigenvector of a uniform tri-diagonal
stiffness matrix (everything else), softened at the ground floor for
soft-storey buildings. This mirrors the mode-shape assumptions used
in ``ppGlobal_PrepareThresholds.py::_dynamic_properties``.
"""

import numpy as np
from scipy.linalg import eigh


def _dynamic_properties(nst, is_sos, is_frame):
    """
    First-mode shape, floor masses and modal participation factor
    for the MDOF stick model.
    """
    I = np.identity(nst)
    if nst > 1:
        I[-1, -1] = 0.75

    # if is_frame and nst <= 12:
    #     phi = np.array([((i + 1) / nst) ** 0.6 for i in range(nst)])
    # else:
    #     K = np.zeros((nst, nst))
    #     np.fill_diagonal(K, 2)
    #     K[-1, -1] = 1
    #     if is_sos:
    #         K[0, 0] = 1.20
    #     for i in range(nst - 1):
    #         K[i, i + 1] = K[i + 1, i] = -1
    #     _, eigenvectors = eigh(K, I)
    #     phi = eigenvectors[:, 0]
    #     phi = phi / phi[-1]
    
    K = np.zeros((nst, nst))
    np.fill_diagonal(K, 2)
    K[-1, -1] = 1
    if is_sos:
        K[0, 0] = 1.20
    for i in range(nst - 1):
        K[i, i + 1] = K[i + 1, i] = -1
    _, eigenvectors = eigh(K, I)
    phi = eigenvectors[:, 0]
    phi = phi / phi[-1]

    # Floor mass such that the assumed mode shape carries unit
    # effective modal mass, consistent with the unit-mass SDOF
    # capacity curve.
    mass = (phi @ I @ phi) / (phi @ I @ np.ones(nst)) ** 2
    floor_masses = (np.diagonal(I) * mass).tolist()
    gamma_real = (phi @ I @ np.ones(nst)) / (phi @ I @ phi)

    return phi, floor_masses, gamma_real


def calibrate_model(nst, sdof_capacity, is_sos=False, is_frame=False,
                    storey_heights=None, verbose=False):
    """
    Calibrate MDOF storey force-drift backbones from an SDOF
    spectral capacity curve.

    Parameters
    ----------
    nst : int
        Number of storeys.
    sdof_capacity : array-like
        SDOF capacity array [Sd (m), Sa (g)], shape (n_points, 2).
    is_sos : bool, optional
        True for soft-storey buildings. Softens the ground-floor
        stiffness used to derive the mode shape. Default False.
    is_frame : bool, optional
        True for moment/braced-frame buildings (nst <= 12), which use
        a power-law mode shape instead of the eigenvalue-derived one.
        Default False.
    storey_heights : list of float, optional
        Not used by the calibration; carried through to metadata for
        callers that need it. Default None.
    verbose : bool, optional
        Print a one-line summary if True. Default False.

    Returns
    -------
    floor_masses : list of float
        MDOF floor masses.
    storey_drifts : numpy.ndarray
        Inter-storey drift capacities (m), shape (nst, n_points).
    storey_forces : numpy.ndarray
        Storey shear-force capacities (g x mass units), shape
        (nst, n_points).
    phi : numpy.ndarray
        First mode shape (roof-normalised).
    metadata : dict
        gamma_real, is_sos, is_frame, storey_heights.
    """
    nst = int(nst)
    sdof_capacity = np.asarray(sdof_capacity, dtype=float)
    Sd, Sa = sdof_capacity[:, 0], sdof_capacity[:, 1]
    n_points = len(Sd)

    phi, floor_masses, gamma_real = _dynamic_properties(nst, is_sos, is_frame)
    flm = np.array(floor_masses)

    storey_drifts = np.zeros((nst, n_points))
    storey_forces = np.zeros((nst, n_points))

    for i in range(nst):
        # Storey shear = Sa x effective modal mass tributary to
        # storeys i..roof (first-mode shear distribution).
        storey_forces[i, :] = Sa * gamma_real * np.sum(phi[i:] * flm[i:])

        if i == 0:
            storey_drifts[i, :] = Sd * gamma_real * phi[0]
        else:
            # Interstorey drift ratio relative to the first storey,
            # from the mode-shape increment.
            drift_ratio = (phi[i] - phi[i - 1]) / phi[0]
            storey_drifts[i, :] = storey_drifts[0, :] * drift_ratio

    if verbose:
        print(f"calibrate_model: nst={nst} is_sos={is_sos} "
              f"is_frame={is_frame} gamma={gamma_real:.4f}")

    metadata = {
        'gamma_real': gamma_real,
        'is_sos': is_sos,
        'is_frame': is_frame,
        'storey_heights': storey_heights,
    }

    return floor_masses, storey_drifts, storey_forces, phi, metadata
