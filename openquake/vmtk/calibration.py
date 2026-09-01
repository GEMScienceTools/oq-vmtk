"""
SDOF-to-MDOF calibration module.

Transforms an SDOF spectral capacity curve ``(Sd, Sa)`` into per-storey
force-drift backbones for a stick-and-mass MDOF model.

The first-mode shape is either an assumed power law (frame buildings up
to 12 storeys) or the first eigenvector of a uniform tri-diagonal
stiffness matrix (everything else), softened at the ground floor for
soft-storey buildings. Soft-storey buildings always use the eigenvector
shape, regardless of ``is_frame``, so the ground-floor softening is never
bypassed.

"""

import numpy as np
from scipy.linalg import eigh


def _validate_inputs(nst, sdof_capacity, is_sos, is_frame, storey_heights):
    """Validate calibrate_model's inputs, raising on the first problem found."""
    if not isinstance(nst, (int, np.integer)) or nst < 1:
        raise ValueError(f"'nst' must be a positive integer, got {nst!r}.")

    sdof_capacity = np.atleast_2d(np.asarray(sdof_capacity, dtype=float))
    if sdof_capacity.ndim != 2 or sdof_capacity.shape[1] != 2:
        raise ValueError(
            "'sdof_capacity' must have shape (n, 2) with columns [Sd, Sa]."
        )
    if sdof_capacity.shape[0] < 2:
        raise ValueError("'sdof_capacity' must have at least 2 points.")
    if np.any(sdof_capacity < 0):
        raise ValueError("'sdof_capacity' values must be non-negative.")

    if not isinstance(is_sos, bool):
        raise TypeError(f"'is_sos' must be a bool, got {type(is_sos).__name__}.")
    if not isinstance(is_frame, bool):
        raise TypeError(f"'is_frame' must be a bool, got {type(is_frame).__name__}.")

    if storey_heights is not None:
        if not hasattr(storey_heights, "__len__"):
            raise TypeError("'storey_heights' must be a list or array.")
        if len(storey_heights) != nst:
            raise ValueError(
                f"'storey_heights' length ({len(storey_heights)}) "
                f"must match 'nst' ({nst})."
            )
        if any(h <= 0 for h in storey_heights):
            raise ValueError("All values in 'storey_heights' must be positive.")


def _dynamic_properties(nst, is_sos, is_frame):
    """
    First-mode shape, floor masses and modal participation factor
    for the MDOF stick model.
    """
    I = np.identity(nst)
    if nst > 1:
        I[-1, -1] = 0.75

    if is_frame and nst <= 12 and not is_sos:
        phi = np.array([((i + 1) / nst) ** 0.6 for i in range(nst)])
    else:
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
        Ignored when is_sos is True: soft-storey buildings always use
        the eigenvalue-derived shape, so the ground-floor softening is
        never bypassed. Default False.
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

    Raises
    ------
    ValueError
        If nst, sdof_capacity, or storey_heights is malformed (see
        above).
    TypeError
        If is_sos, is_frame, or storey_heights has the wrong type.
    """
    _validate_inputs(nst, sdof_capacity, is_sos, is_frame, storey_heights)

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
