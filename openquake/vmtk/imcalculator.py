import numpy as np
from scipy import signal, integrate

# Gravitational acceleration constant (m/s²)
_G = 9.81

# Accepted unit strings for the input acceleration
_VALID_UNITS = {"g", "m/s2", "m/s^2"}


class imcalculator:
    """
    Compute various intensity measures (IMs) from a ground-motion record.

    This class provides functionality to compute response spectra,
    spectral accelerations, amplitude-based intensity measures,
    Arias Intensity, Cumulative Absolute Velocity (CAV), significant
    duration, and the filtered incremental velocity (FIV3) from an
    acceleration time series.

    The input acceleration may be supplied in units of g or m/s².
    Internally, all computations normalise the record to g; the
    ``acc_m_s2`` property provides the record in m/s² at any time.

    Attributes
    ----------
    acc : numpy.ndarray
        The acceleration time series stored internally in g.

    dt : float
        The time step of the accelerogram (s).

    damping : float
        The damping ratio (default is 5%).

    unit : str
        The unit string supplied at construction (``"g"``,
        ``"m/s2"``, or ``"m/s^2"``).

    Methods
    -------
    get_spectrum(periods, damping_ratio)
        Computes the response spectrum using the Newmark-beta method.

    get_sa(period)
        Computes the spectral acceleration at a given period.

    get_saavg(period)
        Computes the geometric mean of spectral accelerations over a
        range of periods centred on the conditioning period.

    get_saavg_user_defined(periods_list)
        Computes the geometric mean of spectral accelerations for a
        user-defined list of periods.

    get_velocity_displacement_history()
        Computes velocity and displacement history with zero-phase
        high-pass filtering and baseline drift correction.

    get_amplitude_ims()
        Computes amplitude-based intensity measures (PGA, PGV, PGD).

    get_arias_intensity()
        Computes the Arias Intensity.

    get_cav()
        Computes the Cumulative Absolute Velocity (CAV).

    get_significant_duration(start, end)
        Computes the significant duration (time between specified
        fractions of Arias intensity).

    get_FIV3(period, alpha, beta)
        Computes the filtered incremental velocity (FIV3).

    get_rotdxx(acc2, percentile, periods, damping_ratio)
        Computes the RotDxx orientation-independent spectral acceleration.

    """

    def __init__(self, acc, dt, damping=0.05, unit="g"):
        """
        Initializes the imcalculator with the input ground-motion
        record.

        The acceleration is converted to g on input so that all
        downstream methods use a consistent unit system. The original
        unit label is stored in ``self.unit`` for reference.

        Parameters
        ----------
        acc : list or numpy.ndarray
            Acceleration time series. Units are specified by the
            ``unit`` parameter.

        dt : float
            Time step of the accelerogram (s).

        damping : float, optional
            Damping ratio (default is 0.05, i.e. 5%).

        unit : str, optional
            Unit of the input acceleration. Accepted values are
            ``"g"`` (default), ``"m/s2"``, or ``"m/s^2"``.

        Raises
        ------
        ValueError
            If ``unit`` is not one of the accepted strings.

        """
        acc = np.array(acc, dtype=float)

        # Validate the unit string
        unit_lower = unit.lower().strip()
        if unit_lower not in _VALID_UNITS:
            raise ValueError(
                f"'unit' must be one of {sorted(_VALID_UNITS)}, "
                f"got '{unit}'."
            )

        # Store acceleration internally in g
        if unit_lower in ("m/s2", "m/s^2"):
            self.acc = acc / _G
        else:
            self.acc = acc

        self.dt = dt
        self.damping = damping
        self.unit = unit_lower

    @property
    def acc_m_s2(self):
        """
        Acceleration time series in m/s².

        Returns
        -------
        numpy.ndarray
            The acceleration record converted to m/s².

        """
        return self.acc * _G

    def get_spectrum(
        self,
        periods=np.linspace(1e-5, 4.0, 500),
        damping_ratio=0.05,
    ):
        """
        Computes the response spectrum using the Newmark-beta method.

        The method performs Newmark constant-average-acceleration
        time integration (gamma = 0.5, beta = 0.25) for a unit-mass
        single-degree-of-freedom oscillator at each requested period,
        returning the spectral displacement, pseudo-velocity, and
        pseudo-acceleration.

        Parameters
        ----------
        periods : numpy.ndarray, optional
            Array of periods at which to compute the spectral response
            (s). Default is 500 points linearly spaced from 1e-5 to
            4.0 s.

        damping_ratio : float, optional
            Damping ratio for the SDOF oscillator. Default is 0.05
            (5%).

        Returns
        -------
        periods : numpy.ndarray
            Periods of the response spectrum (s).

        sd : numpy.ndarray
            Spectral displacement (m).

        sv : numpy.ndarray
            Pseudo spectral velocity (m/s).

        sa : numpy.ndarray
            Pseudo spectral acceleration (g).

        Notes
        -----
        The Newmark-beta parameters used are gamma = 0.5 and
        beta = 0.25, which correspond to the constant average
        acceleration method (unconditionally stable).

        """
        # Newmark-beta integration constants
        gamma = 0.5
        beta = 0.25
        ms = 1.0  # Unit mass (kg)

        # Convert ground acceleration to m/s² and create force vector
        acc = self.acc_m_s2
        p = -ms * acc

        # Number of time steps in the record
        time_steps = len(acc)

        # Initialize response arrays for all periods simultaneously
        num_periods = len(periods)
        u = np.zeros((num_periods, time_steps))  # Displacement
        v = np.zeros((num_periods, time_steps))  # Velocity
        a = np.zeros((num_periods, time_steps))  # Acceleration

        # Compute stiffness, circular frequency, and damping coefficient
        # for all periods at once (vectorised)
        omega = 2 * np.pi / periods  # Circular frequency (rad/s)
        k = ms * omega**2  # Stiffness (N/m)
        c = 2 * damping_ratio * ms * omega  # Damping coefficient

        # Initial acceleration from the first force increment
        a[:, 0] = p[0] / ms

        # Precompute effective stiffness and auxiliary coefficients
        k_bar = (
            k
            + (gamma / (beta * self.dt)) * c
            + (ms / (beta * self.dt**2))
        )
        A = ms / (beta * self.dt) + (gamma / beta) * c
        B = ms / (2 * beta) + (self.dt * c * (gamma / (2 * beta) - 1))

        # Newmark time integration (vectorised over all periods)
        for i in range(time_steps - 1):
            dp = p[i + 1] - p[i]
            dp_bar = dp + A * v[:, i] + B * a[:, i]
            du = dp_bar / k_bar
            dv = (
                (gamma / (beta * self.dt)) * du
                - (gamma / beta) * v[:, i]
                + self.dt * (1 - gamma / (2 * beta)) * a[:, i]
            )
            da = (
                du / (beta * self.dt**2)
                - v[:, i] / (beta * self.dt)
                - a[:, i] / (2 * beta)
            )

            u[:, i + 1] = u[:, i] + du
            v[:, i + 1] = v[:, i] + dv
            a[:, i + 1] = a[:, i] + da

        # Compute spectral values (vectorised across all periods)
        sd = np.max(np.abs(u), axis=1)  # Spectral displacement (m)
        sv = sd * omega  # Pseudo spectral velocity (m/s)
        sa = sd * omega**2 / _G  # Pseudo spectral acceleration (g)

        return periods, sd, sv, sa

    def get_sa(self, period):
        """
        Computes spectral acceleration at a given period by
        interpolating the full response spectrum.

        Parameters
        ----------
        period : float
            The target period (s).

        Returns
        -------
        sa_interp : float
            Spectral acceleration (g) at the requested period.

        """
        periods, _, _, sa = self.get_spectrum()

        # Interpolate to find SA at the requested period
        return np.interp(period, periods, sa)

    def get_saavg(self, period):
        """
        Computes the geometric mean of spectral accelerations over a
        range of periods centred on a conditioning period.

        The period range spans from 0.2 * period to 1.5 * period,
        sampled at 10 equally spaced points.

        Parameters
        ----------
        period : float
            Conditioning period (s).

        Returns
        -------
        sa_avg : float
            Geometric mean of spectral accelerations (g) over the
            defined period range.

        """
        periods, _, _, sa = self.get_spectrum()

        # Define 10 equally spaced periods in [0.2T, 1.5T]
        period_range = np.linspace(0.2 * period, 1.5 * period, 10)

        # Interpolate SA values at the defined period range
        sa_values = np.interp(period_range, periods, sa)

        # Clip to prevent underflow in the log-space geometric mean
        sa_values = np.clip(sa_values, 1e-6, None)

        # Geometric mean via log-space averaging
        return np.exp(np.mean(np.log(sa_values)))

    def get_saavg_user_defined(self, periods_list):
        """
        Computes the geometric mean of spectral accelerations for a
        user-defined list of periods.

        Parameters
        ----------
        periods_list : list or numpy.ndarray
            List of user-defined periods (s) at which spectral
            accelerations are computed.

        Returns
        -------
        sa_avg : float
            Geometric mean of spectral accelerations (g) over the
            user-defined periods.

        """
        periods, _, _, sa = self.get_spectrum()

        # Interpolate SA values at user-defined periods
        sa_values = np.interp(periods_list, periods, sa)

        # Clip to prevent underflow in the log-space geometric mean
        sa_values = np.clip(sa_values, 1e-6, None)

        # Geometric mean via log-space averaging
        return np.exp(np.mean(np.log(sa_values)))

    def get_velocity_displacement_history(self):
        """
        Computes velocity and displacement time histories with
        baseline drift correction.

        A zero-phase fourth-order Butterworth high-pass filter
        (corner frequency 0.1 Hz) is applied to the acceleration
        record before integration. Velocity and displacement are
        obtained by trapezoidal integration of the filtered
        acceleration, with linear detrending applied after each
        integration step to remove residual drift.

        Parameters
        ----------
        None

        Returns
        -------
        vel : numpy.ndarray
            Velocity time history (m/s).

        disp : numpy.ndarray
            Displacement time history (m).

        Notes
        -----
        The zero-phase filter (``sosfiltfilt``) applies the
        Butterworth filter in both the forward and backward
        directions, eliminating phase distortion.

        """
        # Acceleration in m/s²
        acc_m_s2 = self.acc_m_s2

        # Apply a zero-phase 4th-order Butterworth high-pass filter
        # (corner at 0.1 Hz) to remove baseline drift without
        # introducing phase distortion
        sos = signal.butter(
            4, 0.1, btype="highpass", fs=1 / self.dt, output="sos"
        )
        acc_filtered = signal.sosfiltfilt(sos, acc_m_s2)

        # Integrate filtered acceleration to obtain velocity
        vel = integrate.cumulative_trapezoid(
            acc_filtered, dx=self.dt, initial=0
        )
        # Remove linear drift from velocity
        vel = signal.detrend(vel, type="linear")

        # Integrate velocity to obtain displacement
        disp = integrate.cumulative_trapezoid(vel, dx=self.dt, initial=0)
        # Remove residual linear drift from displacement
        disp = signal.detrend(disp, type="linear")

        return vel, disp

    def get_amplitude_ims(self):
        """
        Computes amplitude-based intensity measures.

        Peak Ground Acceleration (PGA), Peak Ground Velocity (PGV),
        and Peak Ground Displacement (PGD) are computed from the
        acceleration time series by successive trapezoidal
        integration.

        Parameters
        ----------
        None

        Returns
        -------
        pga : float
            Peak ground acceleration (g).

        pgv : float
            Peak ground velocity (m/s).

        pgd : float
            Peak ground displacement (m).

        """
        # Acceleration in m/s²
        acc_m_s2 = self.acc_m_s2
        # Integrate acceleration to obtain velocity
        vel = integrate.cumulative_trapezoid(
            acc_m_s2, dx=self.dt, initial=0
        )
        # Integrate velocity to obtain displacement
        disp = integrate.cumulative_trapezoid(vel, dx=self.dt, initial=0)

        return (
            np.max(np.abs(self.acc)),
            np.max(np.abs(vel)),
            np.max(np.abs(disp)),
        )

    def get_arias_intensity(self):
        """
        Computes the Arias Intensity of the ground-motion record.

        Arias Intensity is defined as:

            AI = (pi / 2g) * integral(a(t)^2 dt)

        where a(t) is the ground acceleration in m/s².

        Parameters
        ----------
        None

        Returns
        -------
        ai : float
            Arias Intensity (m/s).

        """
        # Acceleration in m/s²
        acc_m_s2 = self.acc_m_s2
        # Cumulative sum of squared acceleration scaled by pi/(2g)
        ai = np.cumsum(acc_m_s2**2) * (np.pi / (2 * _G)) * self.dt
        # Return the final (total) Arias Intensity value
        return ai[-1]

    def get_cav(self):
        """
        Computes the Cumulative Absolute Velocity (CAV).

        CAV is defined as:

            CAV = integral( abs(a(t)) dt )

        where a(t) is the ground acceleration in m/s².

        Parameters
        ----------
        None

        Returns
        -------
        cav : float
            Cumulative Absolute Velocity (m/s).

        """
        # Integrate the absolute acceleration (m/s²) over the full
        # record duration
        cav = np.sum(np.abs(self.acc_m_s2)) * self.dt
        return cav

    def get_significant_duration(self, start=0.05, end=0.95):
        """
        Computes the significant duration of the ground-motion record.

        Significant duration is defined as the elapsed time between
        specified fractions of the normalised Arias Intensity. The
        default thresholds correspond to the 5%-95% significant
        duration (t_5-95).

        Parameters
        ----------
        start : float, optional
            Lower fraction of normalised Arias Intensity. Default is
            0.05 (5%).

        end : float, optional
            Upper fraction of normalised Arias Intensity. Default is
            0.95 (95%).

        Returns
        -------
        sig_duration : float
            Significant duration (s).

        Notes
        -----
        Because the Arias Intensity is normalised, the result is
        independent of the acceleration unit (g or m/s²).

        """
        # Compute cumulative Arias Intensity (un-normalised).
        # Using self.acc (in g) is valid here because the subsequent
        # normalisation cancels the unit factor.
        ai = np.cumsum(self.acc**2) * (np.pi / (2 * _G)) * self.dt
        # Normalise by the total Arias Intensity
        ai_norm = ai / ai[-1]

        # Find the time instants at which the thresholds are exceeded
        t_start = np.searchsorted(ai_norm, start) * self.dt
        t_end = np.searchsorted(ai_norm, end) * self.dt

        return t_end - t_start

    def get_FIV3(self, period, alpha, beta):
        """
        Computes the filtered incremental velocity (FIV3) intensity
        measure for a given ground-motion record.

        FIV3 is computed following Dávalos and Miranda (2019). A
        second-order low-pass Butterworth filter is applied to the
        acceleration record; the filtered incremental velocity (FIV)
        is then obtained by integrating successive alpha*T windows.
        FIV3 is the maximum of the sum of the three largest peaks and
        the absolute sum of the three deepest troughs.

        The FIV computation is fully vectorised using a cumulative-sum
        approach for the sliding-window trapezoidal integrals,
        avoiding the per-window Python loop.

        Parameters
        ----------
        period : float
            The period (s) used to define the filter cut-off frequency
            and integration window length.

        alpha : float
            Period factor defining the integration window length
            (window duration = alpha * period).

        beta : float
            Cut-off frequency factor for the low-pass Butterworth
            filter (f_c = beta / period).

        Returns
        -------
        FIV3 : float
            FIV3 intensity measure (Eq. 3 of Dávalos & Miranda 2019).

        FIV : numpy.ndarray
            Filtered incremental velocity time series (Eq. 2).

        t : numpy.ndarray
            Time instants corresponding to each FIV value (s).

        ugf : numpy.ndarray
            Low-pass-filtered acceleration time history (g).

        pks : numpy.ndarray
            Up to three largest peaks of the FIV series.

        trs : numpy.ndarray
            Up to three deepest troughs of the FIV series.

        References
        ----------
        Dávalos H, Miranda E. "Filtered incremental velocity: A novel
        approach in intensity measures for seismic collapse
        estimation." *Earthquake Engineering & Structural Dynamics*,
        2019, 48(12), 1384-1405.
        DOI: 10.1002/eqe.3205.

        """
        n = len(self.acc)

        # Build time vector (vectorised replacement for list
        # comprehension)
        tim = np.arange(n) * self.dt

        # Apply a 2nd-order Butterworth low-pass filter to the ground
        # motion record with normalised cut-off frequency
        Wn = beta / period / (0.5 / self.dt)
        b, a = signal.butter(2, Wn, "low")
        ugf = signal.filtfilt(b, a, self.acc)

        # Window length in samples
        w = int(np.floor(alpha * period / self.dt))

        # Determine valid starting indices: the remaining record must
        # be at least alpha*T long (strict inequality matches the
        # original loop condition)
        cutoff_time = tim[-1] - alpha * period
        valid_mask = tim < cutoff_time
        valid_idx = np.where(valid_mask)[0]

        # Further restrict so that i + w does not exceed n
        valid_idx = valid_idx[valid_idx + w <= n]

        # Vectorised sliding-window trapezoidal integration.
        # The trapezoidal integral of ugf[i : i+w] with unit spacing
        # is: sum(ugf[i:i+w]) - 0.5*ugf[i] - 0.5*ugf[i+w-1].
        # Multiplying by self.dt converts to physical units.
        cs = np.cumsum(ugf)
        cs = np.concatenate(([0.0], cs))  # cs[k] = sum(ugf[0:k])

        window_sums = cs[valid_idx + w] - cs[valid_idx]
        FIV = self.dt * (
            window_sums
            - 0.5 * ugf[valid_idx]
            - 0.5 * ugf[valid_idx + w - 1]
        )
        t = tim[valid_idx]

        # Find the peaks and troughs of the FIV array
        pks_ind, _ = signal.find_peaks(FIV)
        trs_ind, _ = signal.find_peaks(-FIV)

        # Sort peak and trough values
        pks_srt = np.sort(FIV[pks_ind])
        trs_srt = np.sort(FIV[trs_ind])

        # Extract the three largest peaks and three deepest troughs
        pks = pks_srt[-3:]
        trs = trs_srt[0:3]

        # FIV3 = max of summed peak energy vs summed trough energy.
        # Troughs are negative, so compare absolute values and return
        # the dominant (unsigned) magnitude per Eq. (3) of the paper.
        FIV3 = np.max([np.sum(pks), np.abs(np.sum(trs))])

        return FIV3, FIV, t, ugf, pks, trs

    def get_rotdxx(
        self,
        acc2,
        percentile=50,
        periods=np.linspace(1e-5, 4.0, 500),
        damping_ratio=0.05,
    ):
        """
        Computes the RotDxx orientation-independent spectral
        acceleration from two horizontal ground-motion components.

        RotDxx is the *xx*-th percentile of the single-component
        spectral acceleration computed over 180 equally spaced
        rotation angles (0° to 179°). The rotated acceleration at
        angle θ is:

            a_rot(t, θ) = a₁(t) · cos θ + a₂(t) · sin θ

        where a₁ and a₂ are the two orthogonal horizontal
        components. Because the system is linear, the displacement
        response to a_rot is:

            u(t, θ) = cos θ · u₁(t) + sin θ · u₂(t)

        where u₁ and u₂ are the SDOF displacement responses to a₁
        and a₂ respectively. This allows the Newmark-β integration
        to be performed only twice (once per component) rather than
        180 times.

        Parameters
        ----------
        acc2 : array_like
            Second horizontal acceleration component. Must be the
            same length as ``self.acc`` and supplied in the same
            unit as the first component (the unit used when
            constructing the :class:`imcalculator` instance).

        percentile : float, optional
            Percentile in [0, 100] across rotation angles used to
            define RotDxx. Use 50 for RotD50 (median) or 100 for
            RotD100 (maximum). Default is 50.

        periods : numpy.ndarray, optional
            Array of periods at which to compute RotDxx (s).
            Default is 500 points linearly spaced from 1e-5 to
            4.0 s.

        damping_ratio : float, optional
            Damping ratio for the SDOF oscillator. Default is
            0.05 (5%).

        Returns
        -------
        periods : numpy.ndarray
            Periods of the RotDxx spectrum (s).

        rotdxx : numpy.ndarray
            RotDxx spectral acceleration (g) at each period.

        Notes
        -----
        Common choices are RotD50 (``percentile=50``), which is
        used as the reference IM in ASCE 7-22 ground-motion
        selection, and RotD100 (``percentile=100``), the
        orientation-independent maximum.

        When the second component is zero, RotD100 equals the
        single-component SA and RotD50 equals SA · √2/2 (the
        median of abs(cos θ) over 180 uniformly spaced angles).

        References
        ----------
        Boore, D.M. (2010). "Orientation-independent, nongeometric-
        mean measures of seismic intensity from two horizontal
        components of motion." *Bulletin of the Seismological
        Society of America*, 100(4), 1830–1835.
        DOI: 10.1785/0120090400.

        """
        # Newmark-beta integration constants (constant average
        # acceleration — unconditionally stable)
        gamma_nb = 0.5
        beta_nb = 0.25
        ms = 1.0  # Unit mass (kg)
        dt = self.dt

        # Convert acc2 to g (matching the internal storage of acc1)
        acc2 = np.array(acc2, dtype=float)
        if self.unit in ("m/s2", "m/s^2"):
            acc2_g = acc2 / _G
        else:
            acc2_g = acc2

        # Precompute SDOF system properties (vectorised over periods)
        periods = np.asarray(periods, dtype=float)
        omega = 2 * np.pi / periods          # (n_periods,)
        k = ms * omega**2                    # Stiffness (N/m)
        c = 2 * damping_ratio * ms * omega   # Damping coefficient

        k_bar = (
            k
            + (gamma_nb / (beta_nb * dt)) * c
            + ms / (beta_nb * dt**2)
        )
        A = ms / (beta_nb * dt) + (gamma_nb / beta_nb) * c
        B = ms / (2 * beta_nb) + dt * c * (gamma_nb / (2 * beta_nb) - 1)

        def _newmark_u(acc_g):
            """Return displacement history (n_periods, n_time) for acc_g."""
            acc_ms2 = acc_g * _G
            p = -ms * acc_ms2
            n_time = len(acc_ms2)
            n_per = len(periods)

            u = np.zeros((n_per, n_time))
            v = np.zeros((n_per, n_time))
            a = np.zeros((n_per, n_time))

            a[:, 0] = p[0] / ms

            for i in range(n_time - 1):
                dp = p[i + 1] - p[i]
                dp_bar = dp + A * v[:, i] + B * a[:, i]
                du = dp_bar / k_bar
                dv = (
                    (gamma_nb / (beta_nb * dt)) * du
                    - (gamma_nb / beta_nb) * v[:, i]
                    + dt * (1 - gamma_nb / (2 * beta_nb)) * a[:, i]
                )
                da = (
                    du / (beta_nb * dt**2)
                    - v[:, i] / (beta_nb * dt)
                    - a[:, i] / (2 * beta_nb)
                )
                u[:, i + 1] = u[:, i] + du
                v[:, i + 1] = v[:, i] + dv
                a[:, i + 1] = a[:, i] + da

            return u

        # SDOF displacement responses for the two components
        u1 = _newmark_u(self.acc)   # (n_periods, n_time)
        u2 = _newmark_u(acc2_g)     # (n_periods, n_time)

        # 180 rotation angles: 0°, 1°, …, 179°
        theta_rad = np.deg2rad(np.arange(180))   # (180,)

        # SA at each (angle, period) combination
        # u_rot(θ) = cos(θ)*u1 + sin(θ)*u2  →  shape (n_periods, n_time)
        # Vectorise over angles by broadcasting
        # cos_th: (180, 1, 1), u1: (1, n_periods, n_time)
        cos_th = np.cos(theta_rad)[:, np.newaxis, np.newaxis]  # (180,1,1)
        sin_th = np.sin(theta_rad)[:, np.newaxis, np.newaxis]  # (180,1,1)
        u_rot = cos_th * u1[np.newaxis] + sin_th * u2[np.newaxis]
        # u_rot shape: (180, n_periods, n_time)

        sd_rot = np.max(np.abs(u_rot), axis=2)   # (180, n_periods)
        sa_rot = sd_rot * omega[np.newaxis, :] ** 2 / _G   # (180, n_periods)

        # RotDxx: percentile across the 180 rotation angles
        rotdxx = np.percentile(sa_rot, percentile, axis=0)   # (n_periods,)

        return periods, rotdxx
