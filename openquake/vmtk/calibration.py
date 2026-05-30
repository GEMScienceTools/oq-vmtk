import warnings
import numpy as np
from scipy.linalg import eigh
from openquake.vmtk.modeller import modeller as _modeller

# Gravitational acceleration constant (m/s²)
_G = 9.81

# Building height classification thresholds (number of storeys)
_LOW_RISE_MAX = 3   # 1–3 storeys
_MID_RISE_MAX = 9   # 4–9 storeys


class calibration:
    """
    Calibrate MDOF storey force-deformation relationships from
    SDOF spectral capacity parameters.

    This class encapsulates the complete SDOF-to-MDOF
    transformation pipeline: building classification, mass and
    stiffness matrix assembly, eigenvalue-based modal analysis,
    period scaling, storey force-drift distribution, and
    (optionally) OpenSees SPO verification.

    The calibration uses a stick-and-mass idealisation consistent
    with the ``modeller`` class: each storey is represented by a
    Pinching4 hysteretic spring whose stiffness and strength are
    derived from the input SDOF capacity curve via first-mode
    N2 / Capacity Spectrum relationships.

    The assumptions are fixed for all systems and building heights:

        Uniform ductility  : all storeys share the same displacement
                             backbone. Every storey drift equals the
                             first-storey drift (Sd x Gamma x phi[0]),
                             so only storey forces vary with height
                             via the first-mode shear ratio
                             V_i / V_base.

        Stiffness grouping : adjacent storeys are paired
                             (group_size = 2), reducing the number of
                             independent spring stiffnesses to
                             ceil(nst / 2) and leading to stiffness
                             decay per ``k = 1.0 - 0.30 * h``,
                             min 0.50.

        Default roof mass factor: 0.75.

        No higher-mode correction is applied.  Only the first mode is
        used for the SDOF-to-MDOF transformation.

    Attributes
    ----------
    nst : int
        Number of storeys.
    sdof_capacity : numpy.ndarray
        SDOF capacity array ``[Sd (m), Sa (g)]``, shape
        ``(n_points, 2)``.
    Sd : numpy.ndarray
        Spectral displacement values (m) extracted from
        *sdof_capacity*.
    Sa : numpy.ndarray
        Spectral acceleration values (g) extracted from
        *sdof_capacity*.
    is_sos : bool
        ``True`` if a soft-storey system.
    storey_heights : list or None
        Storey heights in metres (triggers OpenSees
        verification when provided).
    T_target : float
        Target fundamental period (s), derived from the first
        capacity point.
    category : str
        Building height classification (``'low-rise'``,
        ``'mid-rise'``, or ``'high-rise'``).
    roof_mass_factor : float
        Ratio of roof mass to typical floor mass.
    soft_storey_factor : float
        Ground-floor stiffness reduction for soft-storey
        systems.
    stiffness_group_size : int
        Number of storeys per stiffness group.
    validate_results : bool
        Whether capacity curves are validated after
        calibration.
    verbose : bool
        Whether to print progress to console.

    Methods
    -------
    calibrate_model()
        Run the full SDOF-to-MDOF calibration pipeline and
        return ``(floor_masses, storey_drifts, storey_forces,
        phi, metadata)``.

    build_mass_matrix(mass_profile=None)
        Build a diagonal mass matrix normalised to 1.0.

    build_stiffness_matrix(stiffness_profile=None)
        Build a tri-diagonal lateral stiffness matrix.

    compute_modal_properties(M, K, num_modes=None)
        Solve the generalised eigenvalue problem and return
        modal properties.

    compute_storey_distribution_ratios(phi, M)
        Compute shear and drift distribution ratios from the
        first mode shape.

    transform_sdof_to_mdof(Sd, Sa, phi, M, Gamma, M_eff,
            shear_ratios, drift_ratios=None)
        Transform SDOF capacity to MDOF storey force-drift
        curves.

    validate_capacity_curve(drifts, forces)
        Ensure monotonic displacement and positive values.

    mdof_to_sdof(base_shear, roof_disp, Gamma, M_eff,
            phi_roof=1.0)
        Convert MDOF pushover results to SDOF coordinates.

    sdof_to_mdof_global(Sd, Sa, Gamma, M_eff, phi_roof=1.0)
        Convert SDOF capacity to MDOF global response.

    References
    ----------
    .. [1] Fajfar, P. (1999). "Capacity spectrum method based on
           inelastic demand spectra." Earthquake Engng Struct Dyn.

    .. [2] Priestley, M.J.N., Calvi, G.M., and Kowalsky, M.J.
           (2007). "Displacement-Based Seismic Design of
           Structures." IUSS Press.

    .. [3] Lu X, McKenna F, Cheng Q, Xu Z, Zeng X, Mahin SA.
           An open-source framework for regional earthquake loss
           estimation using the city-scale nonlinear time history
           analysis. Earthquake Spectra. 2020;36(2):806-831.
           doi:10.1177/8755293019891724

    """

    def __init__(
        self,
        nst,
        sdof_capacity,
        is_sos=False,
        storey_heights=None,
        stiffness_profile=None,
        mass_profile=None,
        phi_target=None,
        roof_mass_factor=0.75,
        soft_storey_factor=None,
        stiffness_group_size=2,
        validate_results=True,
        verbose=False,
    ):
        """
        Initialise the calibration object and validate inputs.

        Parameters
        ----------
        nst : int
            Number of storeys. Must be a positive integer.

        sdof_capacity : numpy.ndarray
            SDOF capacity array ``[Sd (m), Sa (g)]``, shape
            ``(n, 2)``. Must not include ``(0, 0)`` rows and
            must contain at least 2 non-zero points.

        is_sos : bool, optional
            ``True`` if a soft-storey system. Reduces the
            ground-floor stiffness by *soft_storey_factor*.
            Default is ``False``.

        storey_heights : list of float, optional
            Storey heights in metres, length *nst*. When
            provided, triggers OpenSees period-matching and
            SPO verification inside ``calibrate_model()``.
            Default is ``None``.

        stiffness_profile : numpy.ndarray, optional
            Custom relative storey stiffnesses (length *nst*).
            Overrides the default decay profile. Default is
            ``None``.

        mass_profile : numpy.ndarray, optional
            Custom floor masses (length *nst*, should sum to
            1.0). Overrides the default mass distribution.
            Default is ``None``.

        phi_target : numpy.ndarray, optional
            Custom first mode shape (length *nst*,
            roof-normalised). Overrides the eigenvalue-based
            mode shape. Default is ``None``.

        roof_mass_factor : float, optional
            Ratio of roof mass to typical floor mass. Default
            is 0.75.

        soft_storey_factor : float, optional
            Multiplicative reduction for the ground-floor
            stiffness when *is_sos* is ``True``. If ``None``,
            defaults to 0.35 for soft-storey systems and 0.50
            otherwise.

        stiffness_group_size : int, optional
            Number of storeys per stiffness group. Default
            is 2.

        validate_results : bool, optional
            If ``True``, capacity curves are validated for
            monotonicity and positive values after
            calibration. Default is ``True``.

        verbose : bool, optional
            If ``True``, print calibration progress to
            console. Default is ``False``.

        Raises
        ------
        TypeError
            If any input has an incorrect type.

        ValueError
            If any input has an invalid value or inconsistent
            dimensions.
        """

        # nst check
        if not isinstance(nst, int) or nst < 1:
            raise ValueError(
                "'nst' must be a positive integer."
            )

        # sdof_capacity check
        sdof_capacity = np.atleast_2d(
            np.asarray(sdof_capacity, dtype=float)
        )
        if sdof_capacity.ndim != 2 or sdof_capacity.shape[1] != 2:
            raise ValueError(
                "'sdof_capacity' must have shape (n, 2) with "
                "columns [Sd, Sa]."
            )

        # Strip leading (0, 0) rows
        nonzero = ~(
            (sdof_capacity[:, 0] == 0)
            & (sdof_capacity[:, 1] == 0)
        )
        if not np.all(nonzero):
            n_stripped = int(np.sum(~nonzero))
            warnings.warn(
                f"Stripped {n_stripped} leading (0,0) row(s) "
                f"from sdof_capacity."
            )
            sdof_capacity = sdof_capacity[nonzero]
        if len(sdof_capacity) < 2:
            raise ValueError(
                "'sdof_capacity' must have at least 2 "
                "non-zero points."
            )
        if sdof_capacity[0, 0] <= 0 or sdof_capacity[0, 1] <= 0:
            raise ValueError(
                f"First capacity point must have Sd > 0 and "
                f"Sa > 0. Got Sd={sdof_capacity[0, 0]}, "
                f"Sa={sdof_capacity[0, 1]}."
            )
        if not np.all(sdof_capacity > 0):
            raise ValueError(
                "All values in 'sdof_capacity' must be "
                "positive."
            )

        # is_sos check
        if not isinstance(is_sos, bool):
            raise TypeError(
                f"'is_sos' must be a bool, "
                f"got {type(is_sos).__name__}."
            )

        # storey_heights check
        if storey_heights is not None:
            if not hasattr(storey_heights, '__len__'):
                raise TypeError(
                    "'storey_heights' must be a list or "
                    "array."
                )
            if len(storey_heights) != nst:
                raise ValueError(
                    f"'storey_heights' length "
                    f"({len(storey_heights)}) must match "
                    f"'nst' ({nst})."
                )
            if any(h <= 0 for h in storey_heights):
                raise ValueError(
                    "All values in 'storey_heights' must "
                    "be positive."
                )

        # stiffness_profile check
        if stiffness_profile is not None:
            stiffness_profile = np.asarray(
                stiffness_profile, dtype=float
            )
            if len(stiffness_profile) != nst:
                raise ValueError(
                    f"'stiffness_profile' length "
                    f"({len(stiffness_profile)}) must match "
                    f"'nst' ({nst})."
                )
            if np.any(stiffness_profile <= 0):
                raise ValueError(
                    "All values in 'stiffness_profile' must "
                    "be positive."
                )

        # mass_profile check
        if mass_profile is not None:
            mass_profile = np.asarray(
                mass_profile, dtype=float
            )
            if len(mass_profile) != nst:
                raise ValueError(
                    f"'mass_profile' length "
                    f"({len(mass_profile)}) must match "
                    f"'nst' ({nst})."
                )
            if np.any(mass_profile <= 0):
                raise ValueError(
                    "All values in 'mass_profile' must be "
                    "positive."
                )

        # phi_target check
        if phi_target is not None:
            phi_target = np.asarray(phi_target, dtype=float)
            if len(phi_target) != nst:
                raise ValueError(
                    f"'phi_target' length "
                    f"({len(phi_target)}) must match "
                    f"'nst' ({nst})."
                )

        # roof_mass_factor check
        if not isinstance(roof_mass_factor, (int, float)):
            raise TypeError(
                "'roof_mass_factor' must be a number."
            )
        if roof_mass_factor <= 0 or roof_mass_factor > 1.0:
            raise ValueError(
                "'roof_mass_factor' must be in (0, 1.0]."
            )

        # stiffness_group_size check
        if not isinstance(stiffness_group_size, int):
            raise TypeError(
                "'stiffness_group_size' must be an integer."
            )
        if stiffness_group_size < 1:
            raise ValueError(
                "'stiffness_group_size' must be >= 1."
            )

        # Resolve soft_storey_factor default
        if soft_storey_factor is None:
            soft_storey_factor = 0.35 if is_sos else 0.50

        # Store validated attributes
        self.nst = nst
        self.sdof_capacity = sdof_capacity
        self.Sd = sdof_capacity[:, 0]
        self.Sa = sdof_capacity[:, 1]
        self.is_sos = is_sos
        self.storey_heights = storey_heights
        self.stiffness_profile = stiffness_profile
        self.mass_profile = mass_profile
        self.phi_target = phi_target
        self.roof_mass_factor = roof_mass_factor
        self.soft_storey_factor = soft_storey_factor
        self.stiffness_group_size = stiffness_group_size
        self.validate_results = validate_results
        self.verbose = verbose

        # Derived attributes
        self.T_target = (
            2 * np.pi
            * np.sqrt(self.Sd[0] / (self.Sa[0] * _G))
        )
        self.category = self._classify_building()

    ##################################################################
    #                       INTERNAL HELPERS                          #
    ##################################################################

    def _classify_building(self):
        """
        Classify the building by its number of storeys.

        Returns
        -------
        str
            One of ``'low-rise'``, ``'mid-rise'``, or
            ``'high-rise'``.
        """
        if self.nst <= _LOW_RISE_MAX:
            return "low-rise"
        elif self.nst <= _MID_RISE_MAX:
            return "mid-rise"
        else:
            return "high-rise"

    @staticmethod
    def _storey_stiffness_profile(nst, group_size=1):
        """
        Compute a relative inter-storey stiffness profile.

        The profile follows ``k = 1.0 - 0.30 * h`` with a
        minimum of 0.50, where *h* is the normalised height
        of the group centroid (0 at ground, 1 at roof).
        Sections change every *group_size* storeys.

        Parameters
        ----------
        nst : int
            Number of storeys.

        group_size : int, optional
            Number of storeys per stiffness group. Default
            is 2.

        Returns
        -------
        numpy.ndarray
            Relative stiffness values of length *nst*.
        """
        n_groups = int(np.ceil(nst / group_size))
        k_groups = np.ones(n_groups)
        for g in range(n_groups):
            i_mid = (
                g * group_size + (group_size - 1) / 2.0
            )
            h = i_mid / max(nst - 1, 1)
            k_groups[g] = max(1.0 - 0.30 * h, 0.50)
        return np.array([
            k_groups[min(i // group_size, n_groups - 1)]
            for i in range(nst)
        ])

    ##################################################################
    #                      MATRIX ASSEMBLY                           #
    ##################################################################

    def build_mass_matrix(self, mass_profile=None):
        """
        Build a diagonal mass matrix normalised to a total
        of 1.0.

        The SDOF capacity curve is defined for a unit-mass
        oscillator (1 tonne), so the MDOF mass matrix must
        also sum to 1.0 for the modal transformation to
        remain consistent:

            V_base = Sa x M_eff
            where M_eff = (phi^T M 1)^2 / (phi^T M phi)

        All intermediate floors carry an equal share; the roof
        carries a reduced fraction controlled by
        *roof_mass_factor*:

            m_floor = 1.0 / (nst - 1 + roof_mass_factor)
            m_roof  = m_floor x roof_mass_factor

        For ``nst = 1`` the single floor mass is 1.0 (SDOF).

        Parameters
        ----------
        mass_profile : numpy.ndarray, optional
            If provided, used directly as the diagonal (must
            sum to 1.0; length must equal *nst*). If ``None``,
            uses the instance's *mass_profile* attribute.

        Returns
        -------
        numpy.ndarray
            Diagonal mass matrix of shape ``(nst, nst)``.
        """
        mp = (mass_profile if mass_profile is not None
              else self.mass_profile)
        if mp is not None:
            return np.diag(np.asarray(mp, dtype=float))
        if self.nst == 1:
            return np.array([[1.0]])
        m_floor = 1.0 / (
            self.nst - 1 + self.roof_mass_factor
        )
        masses = np.full(self.nst, m_floor)
        masses[-1] = m_floor * self.roof_mass_factor
        return np.diag(masses)

    def build_stiffness_matrix(self, stiffness_profile=None):
        """
        Build a tri-diagonal lateral stiffness matrix.

        Uses a uniform-decay profile ``k = 1.0 - 0.30 * h``
        (min 0.50) with sections grouped every
        *stiffness_group_size* storeys. Pass
        *stiffness_profile* to override entirely. If *is_sos*
        is ``True``, the ground-floor spring is reduced by
        *soft_storey_factor*.

        Parameters
        ----------
        stiffness_profile : numpy.ndarray, optional
            Custom relative stiffness values (length *nst*).
            Overrides the default decay profile and the
            instance's stored profile if provided.

        Returns
        -------
        numpy.ndarray
            Stiffness matrix of shape ``(nst, nst)``.
        """
        sp = (stiffness_profile
              if stiffness_profile is not None
              else self.stiffness_profile)
        if sp is not None:
            k = np.asarray(sp, dtype=float)
        else:
            k = self._storey_stiffness_profile(
                self.nst, self.stiffness_group_size
            )
            if self.is_sos:
                k[0] *= self.soft_storey_factor

        nst = self.nst
        K = np.zeros((nst, nst))
        for i in range(nst):
            K[i, i] = k[i] + (
                k[i + 1] if i + 1 < nst else 0.0
            )
            if i > 0:
                K[i, i - 1] = -k[i]
                K[i - 1, i] = -k[i]
        return K

    ##################################################################
    #                      MODAL ANALYSIS                            #
    ##################################################################

    @staticmethod
    def compute_modal_properties(M, K, num_modes=None):
        """
        Compute modal properties from mass and stiffness
        matrices.

        Solves the generalised eigenvalue problem
        ``K phi = omega^2 M phi`` and returns periods, mode
        shapes, participation factors, effective masses, and
        mass participation ratios.

        Parameters
        ----------
        M : numpy.ndarray
            Mass matrix, shape ``(nst, nst)``.

        K : numpy.ndarray
            Stiffness matrix, shape ``(nst, nst)``.

        num_modes : int, optional
            Number of modes to compute. Defaults to all.

        Returns
        -------
        dict
            Dictionary containing:

            - ``'eigenvalues'``: omega^2 for each mode.
            - ``'frequencies'``: omega (rad/s).
            - ``'periods'``: T (s).
            - ``'mode_shapes'``: phi matrix
              ``(nst, num_modes)``, each column is a
              roof-normalised mode shape.
            - ``'participation_factors'``: Gamma.
            - ``'effective_masses'``: M*.
            - ``'mass_participation_ratios'``: MPR.
            - ``'cumulative_mass_participation'``:
              cumulative MPR.
            - ``'M_total'``: total mass (trace of M).
        """
        nst = M.shape[0]
        if num_modes is None:
            num_modes = nst
        num_modes = min(num_modes, nst)

        # Solve generalised eigenvalue problem
        eigenvalues, eigenvectors = eigh(K, M)

        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Take only requested modes
        eigenvalues = eigenvalues[:num_modes]
        eigenvectors = eigenvectors[:, :num_modes]

        # Normalise mode shapes: roof = 1.0,
        # first non-zero sign positive
        mode_shapes = np.zeros_like(eigenvectors)
        for i in range(num_modes):
            phi = eigenvectors[:, i]
            if abs(phi[-1]) > 1e-14:
                phi = phi / phi[-1]
            else:
                max_val = np.max(np.abs(phi))
                if max_val > 1e-14:
                    phi = phi / max_val
            for comp in phi:
                if abs(comp) > 1e-10:
                    if comp < 0:
                        phi = -phi
                    break
            mode_shapes[:, i] = phi

        # Compute modal properties
        ones = np.ones(nst)
        M_total = np.trace(M)

        frequencies = np.sqrt(
            np.maximum(eigenvalues, 1e-10)
        )
        periods = 2 * np.pi / frequencies

        participation_factors = np.zeros(num_modes)
        effective_masses = np.zeros(num_modes)
        mass_participation_ratios = np.zeros(num_modes)

        for i in range(num_modes):
            phi = mode_shapes[:, i]
            L_n = phi @ M @ ones
            M_n = phi @ M @ phi

            participation_factors[i] = L_n / M_n
            effective_masses[i] = L_n ** 2 / M_n
            mass_participation_ratios[i] = (
                effective_masses[i] / M_total
            )

        cumulative_mpr = np.cumsum(
            mass_participation_ratios
        )

        return {
            "eigenvalues": eigenvalues,
            "frequencies": frequencies,
            "periods": periods,
            "mode_shapes": mode_shapes,
            "participation_factors": participation_factors,
            "effective_masses": effective_masses,
            "mass_participation_ratios": mass_participation_ratios,
            "cumulative_mass_participation": cumulative_mpr,
            "M_total": M_total,
        }

    ##################################################################
    #                    MODAL DISTRIBUTION                          #
    ##################################################################

    @staticmethod
    def compute_storey_distribution_ratios(phi, M):
        """
        Compute shear and drift distribution ratios from the
        first mode shape.

        Parameters
        ----------
        phi : numpy.ndarray
            First mode shape (roof-normalised), length *nst*.

        M : numpy.ndarray
            Diagonal mass matrix, shape ``(nst, nst)``.

        Returns
        -------
        shear_ratios : numpy.ndarray
            ``V_i / V_base`` for each storey, computed as
            ``sum(m_j phi_j for j >= i) /
            sum(m_j phi_j)``.

        drift_ratios : numpy.ndarray
            ``delta_phi_i / delta_phi_1`` for each storey,
            where ``delta_phi_i = phi[i] - phi[i-1]`` and
            ``delta_phi_1 = phi[0]``.
        """
        nst = len(phi)
        m_diag = np.diag(M)

        # Shear ratios
        total = np.sum(m_diag * phi)
        if abs(total) < 1e-14:
            shear_ratios = np.ones(nst)
        else:
            shear_ratios = np.array([
                np.sum(m_diag[i:] * phi[i:]) / total
                for i in range(nst)
            ])

        # Drift ratios
        delta_phi = np.empty(nst)
        delta_phi[0] = phi[0]
        if nst > 1:
            delta_phi[1:] = phi[1:] - phi[:-1]

        ref = delta_phi[0]
        if abs(ref) < 1e-10:
            warnings.warn(
                "First inter-storey mode-shape increment "
                "is near zero; using uniform drift ratios."
            )
            drift_ratios = np.ones(nst)
        else:
            drift_ratios = delta_phi / ref

        # Ensure drift ratios remain positive
        drift_ratios = np.maximum(drift_ratios, 1e-6)

        return shear_ratios, drift_ratios

    @staticmethod
    def transform_sdof_to_mdof(
        Sd, Sa, phi, M, Gamma, M_eff,
        shear_ratios, drift_ratios=None,
    ):
        """
        Transform SDOF capacity to MDOF storey force-drift
        curves.

        Default behaviour (``drift_ratios=None``):
        uniform-ductility assumption — every storey shares
        the same drift backbone equal to the first-storey
        drift ``Sd x Gamma x phi[0]``.

        When *drift_ratios* are supplied, each storey drift
        is scaled by its ratio relative to the first storey:

            delta_i = delta_1 x drift_ratios[i]

        Parameters
        ----------
        Sd : numpy.ndarray
            Spectral displacement (m).

        Sa : numpy.ndarray
            Spectral acceleration (g).

        phi : numpy.ndarray
            First mode shape (roof-normalised).

        M : numpy.ndarray
            Mass matrix.

        Gamma : float
            Modal participation factor.

        M_eff : float
            Effective modal mass.

        shear_ratios : numpy.ndarray
            Storey shear ratios ``V_i / V_base``.

        drift_ratios : numpy.ndarray, optional
            ``delta_phi_i / delta_phi_1`` for each storey.
            If ``None``, uniform ductility is assumed.

        Returns
        -------
        storey_forces : numpy.ndarray
            Forces in g x mass units ``(nst, n_points)``.

        storey_drifts : numpy.ndarray
            Inter-storey drifts in metres
            ``(nst, n_points)``.
        """
        nst = len(phi)
        n_points = len(Sd)

        if drift_ratios is None:
            drift_ratios = np.ones(nst)

        base_shear = Sa * M_eff
        first_storey_drift = Sd * Gamma * phi[0]

        storey_forces = np.zeros((nst, n_points))
        storey_drifts = np.zeros((nst, n_points))

        for i in range(nst):
            storey_forces[i, :] = (
                base_shear * shear_ratios[i]
            )
            storey_drifts[i, :] = (
                first_storey_drift * drift_ratios[i]
            )

        return storey_forces, storey_drifts

    ##################################################################
    #                        VALIDATION                              #
    ##################################################################

    @staticmethod
    def validate_capacity_curve(drifts, forces):
        """
        Ensure monotonic displacement and positive values.

        If any displacement is not strictly increasing, it is
        nudged to ``1.01 x`` the previous value. All values
        are forced to be positive (absolute value).

        Parameters
        ----------
        drifts : numpy.ndarray
            Storey drift values.

        forces : numpy.ndarray
            Storey force values.

        Returns
        -------
        drifts : numpy.ndarray
            Validated drift values.

        forces : numpy.ndarray
            Validated force values.
        """
        drifts = np.abs(np.asarray(drifts, dtype=float))
        forces = np.abs(np.asarray(forces, dtype=float))

        for i in range(1, len(drifts)):
            if drifts[i] <= drifts[i - 1]:
                drifts[i] = drifts[i - 1] * 1.01

        return drifts, forces

    ##################################################################
    #                   MAIN CALIBRATION METHOD                      #
    ##################################################################

    def calibrate_model(self):
        """
        Run the full SDOF-to-MDOF calibration pipeline.

        The method performs the following steps:

        1. Build mass and stiffness matrices.
        2. Scale the stiffness matrix so that the first
           analytical period matches ``T_target``.
        3. Extract the first mode shape and compute modal
           participation factors.
        4. Compute storey shear and drift distribution ratios.
        5. Rebuild the stiffness matrix from shear ratios for
           mutual consistency with the storey forces.
        6. Transform the SDOF capacity curve to MDOF storey
           force-drift backbones.
        7. Scale storey forces so that spring stiffnesses match
           ``T_target``.
        8. (Optional) If *storey_heights* were provided, build
           an OpenSees model, verify the period, and run a
           static pushover for validation.

        Returns
        -------
        floor_masses : list of float
            Diagonal floor masses from the mass matrix.

        storey_drifts : numpy.ndarray
            Inter-storey drift capacities in metres, shape
            ``(nst, n_points)``.

        storey_forces : numpy.ndarray
            Storey shear-force capacities in g x mass units,
            shape ``(nst, n_points)``.

        phi : numpy.ndarray
            First mode shape (roof-normalised).

        metadata : dict
            Dictionary of calibration metadata including
            ``T_target``, ``Gamma``, ``M_eff``,
            ``shear_ratios``, ``drift_ratios``, mass and
            stiffness matrices, and (when *storey_heights*
            is provided) SPO verification results.
        """
        nst = self.nst
        Sd = self.Sd
        Sa = self.Sa

        # ── Modal matrices ───────────────────────────────
        M = self.build_mass_matrix()
        K = self.build_stiffness_matrix()

        # ── Scale K so T1_analytical = T_target exactly ──
        omega_target = 2 * np.pi / self.T_target
        evals_raw = eigh(K, M, eigvals_only=True)
        lambda_1 = float(np.sort(evals_raw)[0])
        alpha = omega_target ** 2 / lambda_1
        K = K * alpha

        modal_props = self.compute_modal_properties(
            M, K, num_modes=1
        )

        # ── Mode shape ───────────────────────────────────
        if self.phi_target is not None:
            phi = np.asarray(
                self.phi_target, dtype=float
            )
            phi = phi / phi[-1]
            ones = np.ones(nst)
            Gamma = float(
                (phi @ M @ ones) / (phi @ M @ phi)
            )
            M_eff = float(
                (phi @ M @ ones) ** 2
                / (phi @ M @ phi)
            )
        else:
            phi = modal_props["mode_shapes"][:, 0]
            Gamma = float(
                modal_props["participation_factors"][0]
            )
            M_eff = float(
                modal_props["effective_masses"][0]
            )

        MPR_mode1 = M_eff / np.trace(M)
        shear_ratios, drift_ratios = (
            self.compute_storey_distribution_ratios(
                phi, M
            )
        )

        # ── Rebuild K from shear_ratios ──────────────────
        K_SR = np.zeros((nst, nst))
        for i in range(nst):
            K_SR[i, i] = shear_ratios[i] + (
                shear_ratios[i + 1]
                if i + 1 < nst else 0.0
            )
            if i > 0:
                K_SR[i, i - 1] = -shear_ratios[i]
                K_SR[i - 1, i] = -shear_ratios[i]
        lam_SR = float(
            np.sort(
                eigh(K_SR, M, eigvals_only=True)
            )[0]
        )
        alpha = omega_target ** 2 / lam_SR
        K = K_SR * alpha

        # ── SDOF -> MDOF ────────────────────────────────
        storey_forces, storey_drifts = (
            self.transform_sdof_to_mdof(
                Sd, Sa, phi, M, Gamma, M_eff,
                shear_ratios, drift_ratios,
            )
        )

        # ── Scale forces for T_target consistency ────────
        drift0 = float(
            Sd[0] * Gamma * phi[0] * drift_ratios[0]
        )
        base_shear_kN = float(Sa[0] * M_eff * _G)
        k_implied = base_shear_kN / drift0
        k_needed = alpha * float(shear_ratios[0])
        force_scale = k_needed / k_implied
        storey_forces = storey_forces * force_scale

        floor_masses = np.diag(M).tolist()

        if self.validate_results:
            for i in range(nst):
                storey_drifts[i, :], storey_forces[i, :] = (
                    self.validate_capacity_curve(
                        storey_drifts[i, :],
                        storey_forces[i, :],
                    )
                )

        # ── u_roof_target ────────────────────────────────
        u_roof_target = Sd * Gamma * phi[-1]

        # ── Metadata ─────────────────────────────────────
        metadata = {
            "building_category": self.category,
            "num_storeys": nst,
            "T1": modal_props["periods"][0],
            "T_target": self.T_target,
            "Gamma": Gamma,
            "M_eff": M_eff,
            "MPR_mode1": MPR_mode1,
            "shear_ratios": shear_ratios,
            "drift_ratios": drift_ratios,
            "M": M,
            "K": K,
            "stiffness_group_size": self.stiffness_group_size,
            "participation_factors": (
                modal_props["participation_factors"]
            ),
            "effective_masses": modal_props["effective_masses"],
            "mass_participation_ratios": (
                modal_props["mass_participation_ratios"]
            ),
            "cumulative_mass_participation": (
                modal_props["cumulative_mass_participation"]
            ),
            "u_roof_target": u_roof_target,
        }

        if self.storey_heights is None:
            return (
                floor_masses, storey_drifts,
                storey_forces, phi, metadata,
            )

        # ─────────────────────────────────────────────────
        # OpenSees SPO verification
        # ─────────────────────────────────────────────────
        u_roof_ultimate = u_roof_target[-1]

        if self.verbose:
            print("=" * 60)
            print(
                f"calibrate_model  nst={nst}  "
                f"T_target={self.T_target:.4f} s"
            )
            print(
                f"  Gamma={Gamma:.4f}  "
                f"M_eff={M_eff:.4f}  "
                f"alpha={alpha:.4f}"
            )
            print("=" * 60)

        T_opensees = None
        period_error = np.inf
        spo_Sd = None
        spo_Sa = None
        spo_roof_disp = None
        spo_base_shear = None

        try:
            # Unit storey heights for verification model
            _sh_verify = [1.0] * nst

            model = _modeller(
                nst, _sh_verify, floor_masses,
                storey_drifts, storey_forces * _G, False,
            )
            model.compile_model()
            model.do_gravity_analysis()

            # OpenSees eigen solvers enforce N-1 DOF limit.
            # For nst == 1, compute T analytically.
            if nst == 1:
                k11 = float(K[0, 0])
                m11 = float(M[0, 0])
                omega1 = np.sqrt(k11 / m11)
                T_opensees = 2.0 * np.pi / omega1
                period_error = (
                    abs(T_opensees - self.T_target)
                    / self.T_target
                )
            else:
                n_modes = min(nst - 1, 3)
                T_arr, _ = model.do_modal_analysis(
                    num_modes=n_modes,
                    solver='-genBandArpack',
                    plot_modes=False,
                )
                T_opensees = T_arr[0]
                period_error = (
                    abs(T_opensees - self.T_target)
                    / self.T_target
                )

            if self.verbose:
                print(
                    f"  [T]  OpenSees "
                    f"T1={T_opensees:.4f} s  "
                    f"target={self.T_target:.4f} s  "
                    f"err={period_error:.2%}"
                )

            ref_disp = max(u_roof_target[0], 1e-4)
            disp_scale = max(
                (u_roof_ultimate * 1.5) / ref_disp, 2.0
            )
            spo_res = model.do_spo_analysis(
                ref_disp=ref_disp,
                disp_scale_factor=disp_scale,
                push_dir=1,
                phi=phi.tolist(),
                pFlag=False,
            )
            spo_roof_disp = spo_res['spo_disps'][:, -1]
            spo_base_shear = spo_res['spo_rxn']
            spo_Sd = (
                spo_roof_disp / (Gamma * phi[-1])
            )
            spo_Sa = spo_base_shear / (_G * M_eff)
            if self.verbose:
                for k_pt, sd_t in enumerate(Sd):
                    sa_norm = float(
                        np.interp(sd_t, spo_Sd, spo_Sa)
                    )
                    print(
                        f"  [SPO] Pt{k_pt + 1}: "
                        f"Sd={sd_t:.4f}  "
                        f"Sa_spo={sa_norm:.4f}g  "
                        f"Sa_target={Sa[k_pt]:.4f}g"
                    )
        except Exception as e:
            warnings.warn(f"OpenSees SPO failed: {e}")

        metadata.update({
            "alpha": alpha,
            "T_achieved": T_opensees,
            "period_error_final": period_error,
        })
        if spo_Sd is not None:
            metadata.update({
                "spo_sdof_Sd": spo_Sd,
                "spo_sdof_Sa": spo_Sa,
                "spo_roof_disp": spo_roof_disp,
                "spo_base_shear": spo_base_shear,
            })

        return (
            floor_masses, storey_drifts, storey_forces,
            phi, metadata,
        )

    ##################################################################
    #                     UTILITY METHODS                             #
    ##################################################################

    @staticmethod
    def mdof_to_sdof(base_shear, roof_disp, Gamma,
                     M_eff, phi_roof=1.0):
        """
        Convert MDOF pushover results to SDOF coordinates.

        Parameters
        ----------
        base_shear : numpy.ndarray
            Base shear (kN).

        roof_disp : numpy.ndarray
            Roof displacement (m).

        Gamma : float
            Modal participation factor.

        M_eff : float
            Effective modal mass.

        phi_roof : float, optional
            Roof mode-shape value (usually 1.0). Default
            is 1.0.

        Returns
        -------
        Sd : numpy.ndarray
            Spectral displacement (m).

        Sa : numpy.ndarray
            Spectral acceleration (g).
        """
        Sd = roof_disp / (Gamma * phi_roof)
        Sa = base_shear / (_G * M_eff)
        return Sd, Sa

    @staticmethod
    def sdof_to_mdof_global(Sd, Sa, Gamma, M_eff,
                            phi_roof=1.0):
        """
        Convert SDOF capacity to MDOF global response.

        Parameters
        ----------
        Sd : numpy.ndarray
            Spectral displacement (m).

        Sa : numpy.ndarray
            Spectral acceleration (g).

        Gamma : float
            Modal participation factor.

        M_eff : float
            Effective modal mass.

        phi_roof : float, optional
            Roof mode-shape value (usually 1.0). Default
            is 1.0.

        Returns
        -------
        V_base : numpy.ndarray
            Base shear (kN).

        u_roof : numpy.ndarray
            Roof displacement (m).
        """
        V_base = Sa * _G * M_eff
        u_roof = Sd * Gamma * phi_roof
        return V_base, u_roof


def calibrate_model(nst, sdof_capacity, is_sos=False,
                    storey_heights=None, verbose=False, **kwargs):
    """
    Convenience wrapper around :class:`calibration`.

    Instantiates a :class:`calibration` object with the supplied
    parameters and immediately calls its
    :meth:`~calibration.calibrate_model` method.

    Parameters
    ----------
    nst : int
        Number of storeys.

    sdof_capacity : numpy.ndarray
        SDOF capacity array ``[Sd (m), Sa (g)]``, shape ``(n, 2)``.

    is_sos : bool, optional
        ``True`` for soft-storey systems. Default is ``False``.

    storey_heights : list of float, optional
        Storey heights (m). Triggers OpenSees SPO verification
        when provided. Default is ``None``.

    verbose : bool, optional
        Print calibration progress if ``True``. Default is
        ``False``.

    **kwargs
        Additional keyword arguments forwarded to
        :class:`calibration`.

    Returns
    -------
    floor_masses : list of float
        Diagonal floor masses.

    storey_drifts : numpy.ndarray
        Inter-storey drift capacities (m), shape ``(nst, n_points)``.

    storey_forces : numpy.ndarray
        Storey shear-force capacities (g × mass), shape
        ``(nst, n_points)``.

    phi : numpy.ndarray
        First mode shape (roof-normalised).

    metadata : dict
        Calibration metadata.
    """
    cal = calibration(
        nst=nst,
        sdof_capacity=sdof_capacity,
        is_sos=is_sos,
        storey_heights=storey_heights,
        verbose=verbose,
        **kwargs,
    )
    return cal.calibrate_model()
