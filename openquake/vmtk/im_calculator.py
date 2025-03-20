import numpy as np
from scipy import signal, integrate

class IMCalculator:
    """
    A class to compute various intensity measures (IMs) from a ground-motion record, such as response spectrum, 
    spectral acceleration, amplitude-based IMs, Arias Intensity, Cumulative Absolute Velocity (CAV), 
    and significant duration. The class also supports calculating velocity and displacement histories, as well as 
    advanced IMs like the filtered incremental velocity (FIV3).

    Attributes
    ----------
    acc : np.array
        The acceleration time series (m/s² or g).
    dt : float
        The time step of the accelerogram (s).
    damping : float, optional
        The damping ratio (default is 5%).

    Methods
    -------
    __init__(acc, dt, damping=0.05)
        Initializes the IMCalculator with acceleration time series and time step, and optional damping ratio.
    get_spectrum(periods=np.linspace(1e-5, 4.0, 100), damping_ratio=0.05)
        Computes the response spectrum using the Newmark-beta method.
    get_sa(period)
        Computes the spectral acceleration at a given period.
    get_saavg(period)
        Computes the geometric mean of spectral accelerations over a range of periods.
    get_saavg_user_defined(periods_list)
        Computes the geometric mean of spectral accelerations for a user-defined list of periods.
    get_velocity_displacement_history()
        Computes velocity and displacement history with baseline drift correction.
    get_amplitude_ims()
        Computes amplitude-based intensity measures, including PGA, PGV, and PGD.
    get_arias_intensity()
        Computes the Arias Intensity.
    get_cav()
        Computes the Cumulative Absolute Velocity (CAV).
    get_significant_duration(start=0.05, end=0.95)
        Computes the significant duration (time between 5% and 95% of Arias intensity).
    get_duration_ims()
        Computes duration-based intensity measures: Arias Intensity, CAV, and 5%-95% significant duration.
    get_FIV3(period, alpha, beta)
        Computes the filtered incremental velocity (FIV3) based on a ground motion record and specified parameters.
    
    """   
    
    
    def __init__(self, acc, dt, damping=0.05):
        """
        Initialize the intensity measure calculator with the acceleration data and time step.
        
        Parameters
        ----------
        acc : list or np.array
            Acceleration time series (m/s² or g)
        dt : float
            Time step of the accelerogram (s)
        damping : float, optional
            Damping ratio (default is 5%)
        """
        self.acc = np.array(acc)
        self.dt = dt
        self.damping = damping

    def get_spectrum(self, periods=np.linspace(1e-5, 4.0, 100), damping_ratio=0.05):
        """
        Compute the response spectrum using the Newmark-beta method.
    
        Parameters
        ----------
        periods : np.array
            List of periods to compute spectral response (s).
        damping_ratio : float
            Damping ratio (default is 5%).
    
        Returns
        -------
        periods : np.array
            Periods of the response spectrum.
        sd : np.array
            Spectral displacement (m).
        sv : np.array
            Spectral velocity (m/s).
        sa : np.array
            Spectral acceleration (g).
        """
        # Constants
        gamma = 0.5
        beta = 0.25
        ms = 1.0  # Unit mass (kg)
        g = 9.81  # Gravity constant
    
        # Convert ground acceleration to m/s² and create force vector
        acc = np.array(self.acc) * g
        p = -ms * acc
    
        # Time vector
        time_steps = len(acc)
    
        # Initialize response arrays
        num_periods = len(periods)
        u = np.zeros((num_periods, time_steps))  # Displacement
        v = np.zeros((num_periods, time_steps))  # Velocity
        a = np.zeros((num_periods, time_steps))  # Acceleration
    
        # Compute stiffness, frequency, and damping for all periods at once
        omega = 2 * np.pi / periods  # Circular frequency
        k = ms * omega**2  # Stiffness (N/m)
        c = 2 * damping_ratio * ms * omega  # Damping coefficient
    
        # Initial acceleration
        a[:, 0] = p[0] / ms
    
        # Precompute stiffness term for numerical stability
        k_bar = k + (gamma / (beta * self.dt)) * c + (ms / (beta * self.dt**2))
        A = ms / (beta * self.dt) + (gamma / beta) * c
        B = ms / (2 * beta) + (self.dt * c * (gamma / (2 * beta) - 1))
    
        # Newmark time integration (vectorized loop)
        for i in range(time_steps - 1):
            dp = p[i + 1] - p[i]
            dp_bar = dp + A * v[:, i] + B * a[:, i]
            du = dp_bar / k_bar
            dv = (gamma / (beta * self.dt)) * du - (gamma / beta) * v[:, i] + self.dt * (1 - gamma / (2 * beta)) * a[:, i]
            da = du / (beta * self.dt**2) - v[:, i] / (beta * self.dt) - a[:, i] / (2 * beta)
    
            u[:, i + 1] = u[:, i] + du
            v[:, i + 1] = v[:, i] + dv
            a[:, i + 1] = a[:, i] + da
    
        # Compute spectral values (vectorized)
        sd = np.max(np.abs(u), axis=1)  # Spectral displacement
        sv = sd * omega  # Spectral velocity
        sa = sd * omega**2 / g  # Spectral acceleration (normalized by gravity)
        
        return periods, sd, sv, sa

    def get_sa(self, period):
        """
        Get spectral acceleration for a given period.

        Parameters
        ----------
        period : float
            The target period (s)

        Returns
        -------
        sa_interp : float
            Spectral acceleration (g) at the given period
        """
        periods, _, _, sa = self.get_spectrum()
        return np.interp(period, periods, sa)  # Interpolate to find SA at the requested period

    def get_saavg(self, period):
        """
        Compute geometric mean of spectral accelerations over a range of periods.

        Parameters
        ----------
        period : float
            Conditioning period (s)

        Returns
        -------
        sa_avg : float
            Average spectral acceleration at the given period
        """
        periods, _, _, sa = self.get_spectrum()
        period_range = np.linspace(0.2 * period, 1.5 * period, 10)  # Range around the target period

        # Interpolate SA values for the defined period range
        sa_values = np.interp(period_range, periods, sa)

        # Avoid multiplying zero values in geometric mean
        sa_values = np.clip(sa_values, 1e-6, None)  # Prevent underflow

        return np.exp(np.mean(np.log(sa_values)))  # Geometric mean
    
    
    def get_saavg_user_defined(self, periods_list):
        """
        Compute geometric mean of spectral accelerations for user-defined list of periods.

        Parameters
        ----------
        periods_list : list or np.array
            List of user-defined periods (s) for spectral acceleration calculation.

        Returns
        -------
        sa_avg : float
            Geometric mean of spectral accelerations over user-defined periods.
        """
        periods, _, _, sa = self.get_spectrum()

        # Interpolate SA values for user-defined periods
        sa_values = np.interp(periods_list, periods, sa)
        
        # Avoid multiplying zero values in geometric mean
        sa_values = np.clip(sa_values, 1e-6, None)  # Prevent underflow

        return np.exp(np.mean(np.log(sa_values)))  # Geometric mean
        

    def get_velocity_displacement_history(self):
        """
        Compute velocity and displacement history with drift correction.
        
        Returns
        -------
        vel : np.array
            Velocity time-history (m/s)
        disp : np.array
            Displacement time-history (m)
        """
        acc_m_s2 = self.acc * 9.81  # Convert g to m/s² if needed
    
        # High-pass filter to remove baseline drift
        sos = signal.butter(4, 0.1, btype='highpass', fs=1/self.dt, output='sos')
        acc_filtered = signal.sosfilt(sos, acc_m_s2)
    
        # Integrate acceleration to get velocity
        vel = integrate.cumulative_trapezoid(acc_filtered, dx=self.dt, initial=0)
        vel = signal.detrend(vel, type='linear')  # Remove linear drift
    
        # Integrate velocity to get displacement
        disp = integrate.cumulative_trapezoid(vel, dx=self.dt, initial=0)
        disp = signal.detrend(disp, type='linear')  # Remove residual drift
    
        return vel, disp
    
    
    def get_amplitude_ims(self):
        """
        Compute amplitude-based intensity measures.

        Returns
        -------
        pga : float
            Peak ground acceleration (g)
        pgv : float
            Peak ground velocity (m/s)
        pgd : float
            Peak ground displacement (m)
        """
        
        acc_m_s2 = self.acc * 9.81  # Convert g to m/s²
        vel = integrate.cumulative_trapezoid(acc_m_s2, dx=self.dt, initial=0)
        disp = integrate.cumulative_trapezoid(vel, dx=self.dt, initial=0)

        return np.max(np.abs(self.acc)), np.max(np.abs(vel)), np.max(np.abs(disp))

    def get_arias_intensity(self):
        """
        Compute Arias Intensity.
    
        Returns
        -------
        AI : float
            Arias intensity (m/s)
        """
        ai = np.cumsum(self.acc ** 2) * (np.pi / (2 * 9.81)) * self.dt
        return ai[-1]  # Final Arias Intensity value

    def get_cav(self):
        """
        Compute Cumulative Absolute Velocity (CAV).

        Returns
        -------
        CAV : float
            Cumulative absolute velocity (m/s)
        """
        cav = np.sum(np.abs(self.acc)) * self.dt
        return cav

    def get_significant_duration(self, start=0.05, end=0.95):
        """
        Compute significant duration (time between 5% and 95% of Arias intensity).

        Returns
        -------
        sig_duration : float
            Significant duration (s)
        """
        ai = np.cumsum(self.acc ** 2) * (np.pi / (2 * 9.81)) * self.dt
        ai_norm = ai / ai[-1]  # Normalize AI

        t_start = np.where(ai_norm >= start)[0][0] * self.dt
        t_end = np.where(ai_norm >= end)[0][0] * self.dt

        return t_end - t_start
        
    def get_duration_ims(self):
        """
        Compute duration-based intensity measures: Arias Intensity (AI), CAV, and t_595.

        Returns
        -------
        ai : float
            Arias intensity (m/s)
        cav : float
            Cumulative absolute velocity (m/s)
        t_595 : float
            5%-95% significant duration (s)
        """
        ai = self.get_arias_intensity()
        cav = self.get_cav()
        t_595 = self.get_significant_duration()
        return ai, cav, t_595
    
    def get_FIV3(self, period, alpha, beta):
        """
        Computes the filtered incremental velocity (FIV3) intensity measure for a given ground motion record.
    
        This method calculates the FIV3 based on the approach described by Dávalos and Miranda (2019) using a 
        filtered incremental velocity (FIV) method. The FIV3 is an intensity measure that is used in seismic 
        collapse estimation. The method applies a low-pass Butterworth filter to the acceleration time series 
        and then calculates the incremental velocity using a specified period, alpha (period factor), and beta 
        (cut-off frequency factor). 
    
        The FIV3 is computed by considering the three largest peaks and the three smallest troughs of the 
        filtered incremental velocity time series.
    
        Parameters
        ----------
        period : float
            The period (in seconds) used to filter the ground motion record.
        alpha : float
            A period factor that defines the length of the time window used for filtering.
        beta : float
            A cut-off frequency factor that influences the low-pass filter applied to the ground motion record.
    
        Returns
        -------
        FIV3 : float
            The FIV3 intensity measure (as per Equation (3) of Dávalos and Miranda (2019)).
        FIV : np.array
            The filtered incremental velocity time series (as per Equation (2) of Dávalos and Miranda (2019)).
        t : np.array
            The time series corresponding to the FIV values.
        ugf : np.array
            The filtered acceleration time history after applying the low-pass Butterworth filter.
        pks : np.array
            The three largest peak values from the filtered incremental velocity time series, used to compute FIV3.
        trs : np.array
            The three smallest trough values from the filtered incremental velocity time series, used to compute FIV3.
    
        References
        ----------
        Dávalos H, Miranda E. "Filtered incremental velocity: A novel approach in intensity measures for 
        seismic collapse estimation." *Earthquake Engineering & Structural Dynamics*, 2019, 48(12), 1384–1405. 
        DOI: 10.1002/eqe.3205.
        """

        # Import required packages

        # Create the time series of the signal
        tim = [self.dt * i for i in range(len(self.acc))]

        # Apply a 2nd order Butterworth low pass filter to the ground motion
        Wn = beta/period/(0.5/self.dt)
        b, a = signal.butter(2, Wn, 'low')
        ugf = signal.filtfilt(b, a, self.acc)

        # Get the filtered incremental velocity
        FIV = np.array([])
        t = np.array([])
        for i in range(len(tim)):
            # Check if there is enough length in the remaining time series
            if tim[i] < tim[-1] - alpha*period:
                # Get the snippet of the filtered acceleration history
                ugf_pc = ugf[i:i+int(np.floor(alpha*period/self.dt))]

                # Integrate the snippet
                FIV = np.append(FIV, self.dt*integrate.trapezoid(ugf_pc))

                # Log the time
                t = np.append(t, tim[i])

        # Convert
        # Find the peaks and troughs of the FIV array
        pks_ind, _ = signal.find_peaks(FIV)
        trs_ind, _ = signal.find_peaks(-FIV)

        # Sort the values
        pks_srt = np.sort(FIV[pks_ind])
        trs_srt = np.sort(FIV[trs_ind])

        # Get the peaks
        pks = pks_srt[-3:]
        trs = trs_srt[0:3]

        # Compute the FIV3
        FIV3 = np.max([np.sum(pks), np.sum(trs)])

        return FIV3, FIV, t, ugf, pks, trs
