IM Calculator Module
####################

.. module:: im_calculator
   :synopsis: A Python module for computing various intensity measures (IMs) from ground-motion records.

The ``im_calculator`` module provides a class ``IMCalculator`` that calculates
various intensity measures (IMs) from ground-motion records, such as response
spectra, amplitude-based IMs (e.g., peak ground acceleration, peak ground velocity,
peak ground displacement), structure-dependent IMs (e.g., spectral acceleration and
average spectral acceleration), Arias Intensity, Cumulative Absolute Velocity (CAV),
and significant duration. The module also supports the computation of velocity and
displacement histories, as well as advanced IMs like the filtered incremental velocity (FIV3).

Classes
-------

.. class:: IMCalculator(acc, dt, damping=0.05)

   A class to compute various intensity measures (IMs) from a ground-motion record.

   **Attributes**:

   - **acc**: `list` or `np.array`
     The acceleration time series (m/s² or g).
   - **dt**: `float`
     The time step of the accelerogram (s).
   - **damping**: `float`
     The damping ratio (default is 5%).

   :param acc: The acceleration time series (m/s² or g).
   :type acc: list or np.array
   :param dt: The time step of the accelerogram (s).
   :type dt: float
   :param damping: The damping ratio (default is 5%).
   :type damping: float, optional

   .. method:: get_spectrum(periods=np.linspace(1e-5, 4.0, 100), damping_ratio=0.05)

      Computes the response spectrum using the Newmark-beta method.

      :param periods: List of periods to compute spectral response (s).
      :type periods: np.array
      :param damping_ratio: Damping ratio (default is 5%).
      :type damping_ratio: float
      :return: Periods, spectral displacement (m), spectral velocity (m/s), and spectral acceleration (g).
      :rtype: tuple(np.array, np.array, np.array, np.array)

   .. method:: get_sa(period)

      Computes the spectral acceleration at a given period.

      :param period: The target period (s).
      :type period: float
      :return: Spectral acceleration (g) at the given period.
      :rtype: float

   .. method:: get_saavg(period)

      Computes the geometric mean of spectral accelerations over a range of periods.

      :param period: Conditioning period (s).
      :type period: float
      :return: Average spectral acceleration at the given period.
      :rtype: float

   .. method:: get_saavg_user_defined(periods_list)

      Computes the geometric mean of spectral accelerations for a user-defined list of periods.

      :param periods_list: List of user-defined periods (s) for spectral acceleration calculation.
      :type periods_list: list or np.array
      :return: Geometric mean of spectral accelerations over user-defined periods.
      :rtype: float

   .. method:: get_velocity_displacement_history()

      Computes velocity and displacement history with baseline drift correction.

      :return: Velocity time-history (m/s) and displacement time-history (m).
      :rtype: tuple(np.array, np.array)

   .. method:: get_amplitude_ims()

      Computes amplitude-based intensity measures, including PGA, PGV, and PGD.

      :return: Peak ground acceleration (g), peak ground velocity (m/s), and peak ground displacement (m).
      :rtype: tuple(float, float, float)

   .. method:: get_arias_intensity()

      Computes the Arias Intensity.

      :return: Arias intensity (m/s).
      :rtype: float

   .. method:: get_cav()

      Computes the Cumulative Absolute Velocity (CAV).

      :return: Cumulative absolute velocity (m/s).
      :rtype: float

   .. method:: get_significant_duration(start=0.05, end=0.95)

      Computes the significant duration (time between 5% and 95% of Arias intensity).

      :param start: Start threshold for significant duration (default is 5%).
      :type start: float, optional
      :param end: End threshold for significant duration (default is 95%).
      :type end: float, optional
      :return: Significant Duration (s).
      :rtype: float

   .. method:: get_duration_ims()

      Computes duration-based intensity measures: Arias Intensity, CAV, and 5%-95% significant duration.

      :return: Arias Intensity (m/s), Cumulative Absolute Velocity (m/s), and 5%-95% Significant Duration (s).
      :rtype: tuple(float, float, float)

   .. method:: get_FIV3(period, alpha, beta)

      Computes the filtered incremental velocity (FIV3) intensity measure for a given ground motion record.

      :param period: The period (in seconds) used to filter the ground motion record.
      :type period: float
      :param alpha: A period factor that defines the length of the time window used for filtering.
      :type alpha: float
      :param beta: A cut-off frequency factor that influences the low-pass filter applied to the ground motion record.
      :type beta: float
      :return: FIV3 intensity measure, filtered incremental velocity time series, time series, filtered acceleration time history, peaks, and troughs.
      :rtype: tuple(float, np.array, np.array, np.array, np.array, np.array)

References
----------

1. Cordova, P.P., Deierlein, G.G., Mehanny, S.S., and Cornell, C.A. (2000). “Development of
   a two-parameter seismic intensity measure and probabilistic assessment procedure” in
   *Proceedings of the 2nd US–Japan Workshop on Performance-Based Earthquake Engineering
   Methodology for RC Building Structures* (Sapporo, Hokkaido, 2000).

2. Eads, L., Miranda, E., and Lignos, D.G. (2015). "Average spectral acceleration as an
   intensity measure for collapse risk assessment", *Earthquake Engineering and Structural Dynamics*,
   44, 2057–2073. doi: 10.1002/eqe.2575.

3. Kempton, J.J., and Stewart J.P. (2006). "Prediction equations for significant duration
   of earthquake ground motions considering site and near-source effects", *Earthquake Spectra*,
   22(4), 985-1013.

4. Arias, A. (1970). "A measure of earthquake intensity", in *Seismic Design for Nuclear
   Power Plants* (R.J. Hansen, ed.). The MIT Press, Cambridge, MA. 438-483.

5. Dávalos, H. and Miranda, E. (2019). "Filtered incremental velocity: A novel approach
   in intensity measures for seismic collapse estimation." *Earthquake Engineering & Structural Dynamics*,
   48(12), 1384–1405. DOI: 10.1002/eqe.3205.
