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

    get_vel_disp_history(highpass_hz)
        Computes velocity and displacement history with zero-phase
        high-pass filtering (zero-padded) for baseline drift correction.

    get_amplitude_ims(highpass_hz)
        Computes amplitude-based intensity measures (PGA, PGV, PGD);
        PGV and PGD from the drift-corrected histories.

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

    get_rotdxx_amplitude_ims(acc2, percentile, highpass_hz)
        Computes the RotDxx PGA, PGV and PGD.

    get_rotdxx_cav(acc2, percentile)
        Computes the RotDxx Cumulative Absolute Velocity.

    get_rotdxx_arias_duration(acc2, percentile, start, end)
        Computes the RotDxx Arias Intensity and significant duration.

    get_rotdxx_saavg(acc2, period, periods_list, percentile)
        Computes the RotDxx average spectral acceleration.

    get_rotdxx_FIV3(acc2, period, alpha, beta, percentile)
        Computes the RotDxx filtered incremental velocity (FIV3).

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
        damping_ratio=None,
    ):
        """
        Computes the response spectrum using the Newmark-beta method.

        The method performs Newmark constant-average-acceleration
        time integration (gamma = 0.5, beta = 0.25) for a unit-mass
        single-degree-of-freedom oscillator at each requested period,
        returning the spectral displacement, pseudo-spectral velocity
        (psv), and pseudo-spectral acceleration (psa).

        Parameters
        ----------
        periods : numpy.ndarray, optional
            Array of periods at which to compute the spectral response
            (s). Default is 500 points linearly spaced from 1e-5 to
            4.0 s.

        damping_ratio : float, optional
            Damping ratio for the SDOF oscillator. Defaults to
            ``self.damping`` (the damping ratio supplied at
            construction, itself 0.05 / 5% by default).

        Returns
        -------
        periods : numpy.ndarray
            Periods of the response spectrum (s).

        sd : numpy.ndarray
            Spectral displacement (m).

        psv : numpy.ndarray
            Pseudo spectral velocity (m/s).

        psa : numpy.ndarray
            Pseudo spectral acceleration (g).

        Notes
        -----
        The Newmark-beta parameters used are gamma = 0.5 and
        beta = 0.25, which correspond to the constant average
        acceleration method (unconditionally stable).

        """
        if damping_ratio is None:
            damping_ratio = self.damping

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
        psv = sd * omega  # Pseudo spectral velocity (m/s)
        psa = sd * omega**2 / _G  # Pseudo spectral acceleration (g)

        return periods, sd, psv, psa

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
        psa_interp : float
            Pseudo-spectral acceleration (g) at the requested period.

        """
        periods, _, _, psa = self.get_spectrum()

        # Interpolate to find PSA at the requested period
        return np.interp(period, periods, psa)

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
        psa_avg : float
            Geometric mean of pseudo-spectral accelerations (g) over
            the defined period range.

        References
        -------
        Cordova, P., Deierlein, G., Mehanny, S., and Cornell, A., 2000.
            Development of a two-parameter seismic intensity measure and
            probabilistic assessment procedure. 2nd US–Japan Workshop on
            Performance-Based Earthquake Engineering Methodology for RC
            Building Structures.

        Eads, L., Miranda, E., and Lignos, D. G., 2015. Average spectral
            acceleration as an intensity measure for collapse risk
            assessment. Earthquake Engineering & Structural Dynamics,
            44(12), 2057–2073. DOI: 10.1002/eqe.2575

        """
        periods, _, _, psa = self.get_spectrum()

        # Define 10 equally spaced periods in [0.2T, 1.5T]
        period_range = np.linspace(0.2 * period, 1.5 * period, 10)

        # Interpolate PSA values at the defined period range
        psa_values = np.interp(period_range, periods, psa)

        # Clip to prevent underflow in the log-space geometric mean
        psa_values = np.clip(psa_values, 1e-6, None)

        # Geometric mean via log-space averaging
        return np.exp(np.mean(np.log(psa_values)))

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
        psa_avg : float
            Geometric mean of pseudo-spectral accelerations (g) over
            the user-defined periods.

        References
        -------
        Cordova, P., Deierlein, G., Mehanny, S., and Cornell, A., 2000.
            Development of a two-parameter seismic intensity measure and
            probabilistic assessment procedure. 2nd US–Japan Workshop on
            Performance-Based Earthquake Engineering Methodology for RC
            Building Structures.

        Eads, L., Miranda, E., and Lignos, D. G., 2015. Average spectral
            acceleration as an intensity measure for collapse risk
            assessment. Earthquake Engineering & Structural Dynamics,
            44(12), 2057–2073. DOI: 10.1002/eqe.2575

        """
        periods, _, _, psa = self.get_spectrum()

        # Interpolate PSA values at user-defined periods
        psa_values = np.interp(periods_list, periods, psa)

        # Clip to prevent underflow in the log-space geometric mean
        psa_values = np.clip(psa_values, 1e-6, None)

        # Geometric mean via log-space averaging
        return np.exp(np.mean(np.log(psa_values)))

    def _integrate_vel_disp(self, highpass_hz):
        """
        Velocity and displacement histories by trapezoidal integration,
        optionally after a zero-phase high-pass filter with zero padding.

        Parameters
        ----------
        highpass_hz : float or None
            Corner frequency (Hz) of the zero-phase fourth-order
            Butterworth high-pass filter. ``None`` integrates the record
            as it is (no drift correction).

        Returns
        -------
        vel : numpy.ndarray
            Velocity time history (m/s).

        disp : numpy.ndarray
            Displacement time history (m).

        Notes
        -----
        The record is padded with zeros at both ends before filtering
        and the padding is removed after integration, so the filter
        transients and the integration constants do not distort the
        record (Boore, 2005). Each pad is ``1.5 * order / highpass_hz``
        seconds long.

        References
        ----------
        Boore, D. M., 2005. On pads and filters: Processing strong-motion
            data. Bulletin of the Seismological Society of America,
            95(2), 745-750. DOI: 10.1785/0120040160

        """
        acc_m_s2 = self.acc_m_s2
        if highpass_hz is None:
            vel = integrate.cumulative_trapezoid(acc_m_s2, dx=self.dt, initial=0)
            disp = integrate.cumulative_trapezoid(vel, dx=self.dt, initial=0)
            return vel, disp

        nyquist = 0.5 / self.dt
        if not 0 < highpass_hz < nyquist:
            raise ValueError(
                f"'highpass_hz' must be between 0 and the Nyquist "
                f"frequency ({nyquist:g} Hz), got {highpass_hz}"
            )
        order = 4
        npad = int(np.ceil(1.5 * order / highpass_hz / self.dt))
        padded = np.concatenate([np.zeros(npad), acc_m_s2, np.zeros(npad)])

        # Zero-phase fourth-order Butterworth high-pass filter
        sos = signal.butter(
            order, highpass_hz, btype="highpass", fs=1 / self.dt, output="sos"
        )
        acc_filtered = signal.sosfiltfilt(sos, padded)

        # Integrate the padded record, then remove the pads
        vel = integrate.cumulative_trapezoid(acc_filtered, dx=self.dt, initial=0)
        disp = integrate.cumulative_trapezoid(vel, dx=self.dt, initial=0)
        return vel[npad:-npad], disp[npad:-npad]

    def get_vel_disp_history(self, highpass_hz=0.05):
        """
        Computes velocity and displacement time histories with
        baseline drift correction.

        A zero-phase fourth-order Butterworth high-pass filter is
        applied to the zero-padded acceleration record before
        trapezoidal integration (Boore, 2005); the pads are removed
        afterwards.

        Parameters
        ----------
        highpass_hz : float or None, optional
            Corner frequency (Hz) of the high-pass filter. Default is
            0.05 Hz. Use the record's own usable-frequency limit when
            it is known (e.g. from a flatfile). ``None`` integrates the
            record without drift correction.

        Returns
        -------
        vel : numpy.ndarray
            Velocity time history (m/s).

        disp : numpy.ndarray
            Displacement time history (m).

        """
        return self._integrate_vel_disp(highpass_hz)

    def get_amplitude_ims(self, highpass_hz=0.05):
        """
        Computes amplitude-based intensity measures.

        Peak Ground Acceleration (PGA) is the peak of the record. Peak
        Ground Velocity (PGV) and Peak Ground Displacement (PGD) are the
        peaks of the drift-corrected velocity and displacement histories
        of ``get_vel_disp_history``.

        Parameters
        ----------
        highpass_hz : float or None, optional
            Corner frequency (Hz) of the high-pass filter applied before
            integration. Default is 0.05 Hz. ``None`` integrates the
            record without drift correction, which inflates PGD (and to
            a lesser extent PGV) through baseline drift.

        Returns
        -------
        pga : float
            Peak ground acceleration (g).

        pgv : float
            Peak ground velocity (m/s).

        pgd : float
            Peak ground displacement (m).

        """
        vel, disp = self._integrate_vel_disp(highpass_hz)
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

        References
        -------
        Arias, A., 1970. A measure of earthquake intensity. 
            Hansen, R. J. (ed.), Seismic Design for Nuclear Power 
            Plants (pp. 438–483). Cambridge, MA: MIT Press.

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

        References
        -------
        O’Hara, T. F., and Jacobson, J. P., 1991. Standardization
            of the cumulative absolute velocity (EPRI-TR--100082; 
            ON: UN92004453). Palo Alto, CA.


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

        References
        -------
        Trifunac, M. D., and Brady, A. G., 1975. A study on the duration
            of strong earthquake ground motion. Bulletin of the 
            Seismological Society of America, 65(3), 581–626.

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
        Davalos, H., and Miranda, E., 2019. Filtered incremental 
            velocity: A novel approach in intensity measures for 
            seismic collapse estimation. Earthquake Engineering & 
            Structural Dynamics, 48(12), 1384–1405. 
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

        # Three largest peaks / deepest troughs and their FIV3 (Eq. 3)
        FIV3, pks, trs = self._fiv3_from_series(FIV)

        return FIV3, FIV, t, ugf, pks, trs

    @staticmethod
    def _fiv3_from_series(fiv):
        """FIV3, three largest peaks and three deepest troughs of a
        filtered incremental velocity series."""
        pks_ind, _ = signal.find_peaks(fiv)
        trs_ind, _ = signal.find_peaks(-fiv)

        # Extract the three largest peaks and three deepest troughs
        pks = np.sort(fiv[pks_ind])[-3:]
        trs = np.sort(fiv[trs_ind])[0:3]

        # FIV3 = max of summed peak energy vs summed trough energy.
        # Troughs are negative, so compare absolute values and return
        # the dominant (unsigned) magnitude per Eq. (3) of the paper.
        return np.max([np.sum(pks), np.abs(np.sum(trs))]), pks, trs

    def get_rotdxx(
        self,
        acc2,
        percentile=50,
        periods=np.linspace(1e-5, 4.0, 500),
        damping_ratio=None,
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
            Damping ratio for the SDOF oscillator. Defaults to
            ``self.damping`` (the damping ratio supplied at
            construction, itself 0.05 / 5% by default).

        Returns
        -------
        periods : numpy.ndarray
            Periods of the RotDxx spectrum (s).

        rotdxx : numpy.ndarray
            RotDxx pseudo-spectral acceleration (g) at each period.

        Notes
        -----
        Common choices are RotD50 (``percentile=50``), which is
        used as the reference IM in ASCE 7-22 ground-motion
        selection, and RotD100 (``percentile=100``), the
        orientation-independent maximum.

        When the second component is zero, RotD100 equals the
        single-component PSA and RotD50 equals PSA · √2/2 (the
        median of abs(cos θ) over 180 uniformly spaced angles).

        References
        ----------
        Boore, D.M. (2010). "Orientation-independent, nongeometric-
        mean measures of seismic intensity from two horizontal
        components of motion." *Bulletin of the Seismological
        Society of America*, 100(4), 1830–1835.
        DOI: 10.1785/0120090400.

        """
        periods, psa_rot = self._rotated_psa(acc2, periods, damping_ratio)
        # RotDxx: percentile across the 180 rotation angles
        return periods, np.percentile(psa_rot, percentile, axis=0)

    # ------------------------------------------------------------------
    # RotDxx versions of the other intensity measures
    # ------------------------------------------------------------------
    @staticmethod
    def _rot_cos_sin():
        """cos and sin of the 180 rotation angles 0°, 1°, ..., 179°,
        as (180, 1) columns."""
        th = np.deg2rad(np.arange(180))[:, np.newaxis]
        return np.cos(th), np.sin(th)

    def _second_component(self, acc2):
        """Validate the second horizontal component and return an
        imcalculator for it (same dt and damping, stored in g)."""
        acc2 = np.array(acc2, dtype=float)
        if acc2.shape != self.acc.shape:
            raise ValueError(
                "'acc2' must have the same length as the first "
                f"component ({len(self.acc)}), got {len(acc2)}."
            )
        return imcalculator(acc2, self.dt, self.damping, self.unit)

    def _rotate(self, x1, x2):
        """Rotate a linear pair of series to the 180 angles:
        x(θ) = x1·cos θ + x2·sin θ, shape (180, n_time)."""
        ct, st = self._rot_cos_sin()
        return ct * x1[np.newaxis, :] + st * x2[np.newaxis, :]

    def _newmark_disp(self, acc_g, periods, damping_ratio):
        """SDOF displacement histories (n_periods, n_time) and the
        circular frequencies for acceleration ``acc_g`` (in g), by
        constant-average-acceleration Newmark integration of all the
        periods at once."""
        gamma_nb, beta_nb, ms, dt = 0.5, 0.25, 1.0, self.dt
        omega = 2 * np.pi / periods
        c = 2 * damping_ratio * ms * omega
        k_bar = (ms * omega**2 + (gamma_nb / (beta_nb * dt)) * c
                 + ms / (beta_nb * dt**2))
        A = ms / (beta_nb * dt) + (gamma_nb / beta_nb) * c
        B = ms / (2 * beta_nb) + dt * c * (gamma_nb / (2 * beta_nb) - 1)
        p = -ms * acc_g * _G
        u = np.zeros((len(periods), len(p)))
        v = np.zeros(len(periods))
        a = np.full(len(periods), p[0] / ms)
        for i in range(len(p) - 1):
            dp_bar = p[i + 1] - p[i] + A * v + B * a
            du = dp_bar / k_bar
            dv = ((gamma_nb / (beta_nb * dt)) * du
                  - (gamma_nb / beta_nb) * v
                  + dt * (1 - gamma_nb / (2 * beta_nb)) * a)
            da = du / (beta_nb * dt**2) - v / (beta_nb * dt) \
                - a / (2 * beta_nb)
            u[:, i + 1] = u[:, i] + du
            v, a = v + dv, a + da
        return u, omega

    def _rotated_psa(self, acc2, periods, damping_ratio=None):
        """PSA (g) of the rotated record at each (angle, period), shape
        (180, n_periods). Because the oscillator is linear, Newmark is
        run once per component and the displacement histories are
        rotated: u(θ) = cos θ · u1 + sin θ · u2."""
        if damping_ratio is None:
            damping_ratio = self.damping
        periods = np.asarray(periods, dtype=float)
        acc2_g = self._second_component(acc2).acc
        u1, omega = self._newmark_disp(self.acc, periods, damping_ratio)
        u2, _ = self._newmark_disp(acc2_g, periods, damping_ratio)
        ct, st = self._rot_cos_sin()
        psa = np.empty((180, len(periods)))
        for j in range(len(periods)):
            sd = np.max(np.abs(ct * u1[j] + st * u2[j]), axis=1)
            psa[:, j] = sd * omega[j] ** 2 / _G
        return periods, psa

    def get_rotdxx_amplitude_ims(self, acc2, percentile=50,
                                 highpass_hz=0.05):
        """
        Computes the RotDxx PGA, PGV and PGD.

        Acceleration, velocity and displacement are linear in the
        record, so the drift-corrected histories of each component
        (see ``get_vel_disp_history``) are rotated to the 180 angles
        0°, 1°, ..., 179°; the peak absolute value is taken at every
        angle and the ``percentile`` of the 180 peaks is returned.

        Parameters
        ----------
        acc2 : array_like
            Second horizontal component, same length and unit as the
            first one.

        percentile : float, optional
            Percentile across angles (50 = RotD50, 100 = RotD100).

        highpass_hz : float or None, optional
            Corner frequency (Hz) of the drift-correcting high-pass
            filter, as in ``get_amplitude_ims``.

        Returns
        -------
        pga : float
            RotDxx peak ground acceleration (g).

        pgv : float
            RotDxx peak ground velocity (m/s).

        pgd : float
            RotDxx peak ground displacement (m).

        """
        other = self._second_component(acc2)
        v1, d1 = self._integrate_vel_disp(highpass_hz)
        v2, d2 = other._integrate_vel_disp(highpass_hz)
        out = []
        for x1, x2 in ((self.acc, other.acc), (v1, v2), (d1, d2)):
            peaks = np.max(np.abs(self._rotate(x1, x2)), axis=1)
            out.append(np.percentile(peaks, percentile))
        return tuple(out)

    def get_rotdxx_cav(self, acc2, percentile=50):
        """
        Computes the RotDxx Cumulative Absolute Velocity.

        CAV is not linear in the record, so it is recomputed on the
        rotated accelerogram at each of the 180 angles and the
        ``percentile`` across angles is returned.

        Parameters
        ----------
        acc2 : array_like
            Second horizontal component, same length and unit as the
            first one.

        percentile : float, optional
            Percentile across angles (50 = RotD50, 100 = RotD100).

        Returns
        -------
        cav : float
            RotDxx cumulative absolute velocity (m/s).

        """
        other = self._second_component(acc2)
        acc_rot = self._rotate(self.acc_m_s2, other.acc_m_s2)
        cav = np.sum(np.abs(acc_rot), axis=1) * self.dt
        return np.percentile(cav, percentile)

    def get_rotdxx_arias_duration(self, acc2, percentile=50, start=0.05,
                                  end=0.95):
        """
        Computes the RotDxx Arias Intensity and significant duration.

        The rotated cumulative Arias Intensity is a quadratic form of
        the two components, so it is assembled from the cumulative sums
        of a1², a2² and a1·a2 at each of the 180 angles (no rotated
        record is built). The ``percentile`` across angles of the final
        Arias Intensity and of the time between the ``start`` and
        ``end`` fractions of it is returned.

        Parameters
        ----------
        acc2 : array_like
            Second horizontal component, same length and unit as the
            first one.

        percentile : float, optional
            Percentile across angles (50 = RotD50, 100 = RotD100).

        start, end : float, optional
            Fractions of the normalised Arias Intensity defining the
            duration. Default is 0.05 and 0.95 (D5-95).

        Returns
        -------
        ai : float
            RotDxx Arias Intensity (m/s).

        duration : float
            RotDxx significant duration (s).

        """
        other = self._second_component(acc2)
        a1, a2 = self.acc_m_s2, other.acc_m_s2
        ct, st = self._rot_cos_sin()
        energy = (ct**2 * np.cumsum(a1 * a1) + st**2 * np.cumsum(a2 * a2)
                  + 2 * ct * st * np.cumsum(a1 * a2))
        ai = energy[:, -1] * (np.pi / (2 * _G)) * self.dt
        norm = energy / energy[:, -1:]
        # first sample at or above each fraction (as np.searchsorted)
        i_start = np.argmax(norm >= start, axis=1)
        i_end = np.argmax(norm >= end, axis=1)
        dur = (i_end - i_start) * self.dt
        return np.percentile(ai, percentile), np.percentile(dur, percentile)

    def get_rotdxx_saavg(self, acc2, period=None, periods_list=None,
                         percentile=50, damping_ratio=None):
        """
        Computes the RotDxx average spectral acceleration.

        The geometric mean of the spectral accelerations is taken at
        every rotation angle (not of the RotDxx spectrum), and the
        ``percentile`` of the 180 values is returned, i.e. RotDxx of
        the AvgSA definition. The oscillators are integrated at the
        exact periods (no interpolation on a spectrum grid).

        Parameters
        ----------
        acc2 : array_like
            Second horizontal component, same length and unit as the
            first one.

        period : float, optional
            Conditioning period (s): AvgSA over 10 equally spaced
            periods in [0.2*period, 1.5*period], as in ``get_saavg``.

        periods_list : array_like, optional
            User-defined periods (s), as in ``get_saavg_user_defined``.
            Takes precedence over ``period``.

        percentile : float, optional
            Percentile across angles (50 = RotD50, 100 = RotD100).

        damping_ratio : float, optional
            Oscillator damping. Defaults to ``self.damping``.

        Returns
        -------
        saavg : float
            RotDxx average spectral acceleration (g).

        """
        if periods_list is None:
            if period is None:
                raise ValueError("Provide 'period' or 'periods_list'.")
            periods_list = np.linspace(0.2 * period, 1.5 * period, 10)
        _, psa = self._rotated_psa(acc2, periods_list, damping_ratio)
        # Clip to prevent underflow in the log-space geometric mean
        psa = np.clip(psa, 1e-6, None)
        gmean = np.exp(np.mean(np.log(psa), axis=1))
        return np.percentile(gmean, percentile)

    def get_rotdxx_FIV3(self, acc2, period, alpha, beta, percentile=50):
        """
        Computes the RotDxx filtered incremental velocity (FIV3).

        The FIV series (low-pass filter plus window integral) is linear
        in the record, so the series of the two components are rotated
        and only the peak/trough picking of ``get_FIV3`` is repeated at
        each of the 180 angles.

        Parameters
        ----------
        acc2 : array_like
            Second horizontal component, same length and unit as the
            first one.

        period, alpha, beta : float
            As in ``get_FIV3``.

        percentile : float, optional
            Percentile across angles (50 = RotD50, 100 = RotD100).

        Returns
        -------
        FIV3 : float
            RotDxx FIV3 (g·s).

        """
        other = self._second_component(acc2)
        fiv1 = self.get_FIV3(period, alpha, beta)[1]
        fiv2 = other.get_FIV3(period, alpha, beta)[1]
        vals = [self._fiv3_from_series(f)[0]
                for f in self._rotate(fiv1, fiv2)]
        return np.percentile(vals, percentile)
