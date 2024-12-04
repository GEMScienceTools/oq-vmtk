import numpy as np
import eqsig 
from scipy import signal, integrate

class im_calculator():
    
    def __init__(self, acc, dt, damping = 0.05):
        """
        Get intensity measures from a ground-motion record
        Parameters
        ----------
        acc: List
            Accelerations
        dt: float
            Time step of the accelerogram
        damping: float
            Damping
        """
        self.acc = acc
        self.dt  = dt
        self.damping = damping
        
    def get_spectrum(self):
        """
        Get response spectrum of a ground-motion record
        Parameters
        ----------
        acc: List
            Accelerations
        dt: float
            Time step of the accelerogram
        damping: float
            Damping        
        Returns
        -------
        record.response_times: list of periods
        record.s_a: spectral accelerations
        """
        periods = np.linspace(0.0, 4.0, 100)  # compute the response for 100 periods between T=0.0s and 4.0s
        record = eqsig.AccSignal(self.acc, self.dt)
        record.generate_response_spectrum(response_times=periods)
                
        return record.response_times, record.s_a
            
    def get_sa(self, prd, sas, period):
        """
        Get spectral acceleration for a period value
        Parameters
        ----------
        prd: list
            List of periods (obtained from get_spectrum method)
        sas: List
            List of spectral acceleration values (obtained from get_spectrum method)
        period: Float
            Conditioning period (for PGA, period = 0.0)
        
        Returns
        -------
        sa: float
            Spectral acceleration at the given period
        """
        
        ### Return the interpolated spectral acceleration value
        return np.interp(period, prd, sas)
    
    def get_saavg(self, prd, sas, period):
        """
        Get average spectral acceleration for a range of periods
        Parameters
        ----------
        prd: list
            List of periods (obtained from get_spectrum method)
        sas: List
            List of spectral acceleration values (obtained from get_spectrum method)
        period: Float
            Conditioning period
        
        Returns
        -------
        saavg: float
            Average spectral acceleration at the given period
        """        
        
        ### Define the period range
        period_range = np.linspace(0.2*period,1.5*period, 10)
        
        ### Interpolate for spectral acceleration values at each period within the range
        sa = [np.interp(i,prd,sas) for i in period_range]
        
        ### Return the average spectral acceleration
        return np.prod(sa)**(1/10)
    
    def get_FIV3(self, period, alpha, beta):
        """
        References:
        Dávalos H, Miranda E. Filtered incremental velocity: A novel approach in intensity measures for
        seismic collapse estimation. Earthquake Engineering & Structural Dynamics 2019; 48(12): 1384–1405.
        DOI: 10.1002/eqe.3205.

        Get the filtered incremental velocity IM for a ground motion
        Parameters
        ----------
            Period:     Float
            Period [s]
            alpha:      Float
            Period factor (see Figure 6)
            beta:       Float
            Cut-off frequency factor (see Figure 6)

        Returns:
            FIV3:       Intensity measure FIV3 (as per Eq. (3) of Davalos and Miranda (2019))
            FIV:        Filtered incremental velocity (as per Eq. (2) of Davalos and Miranda (2019))
            t:          Time series of FIV
            ugf:        Filtered acceleration time history
            pks:        Three peak values used to compute FIV3
            trs:        Three trough values used to compute FIV3
        """
        # Import required packages

        # Create the time series of the signal
        tim = self.dt*range(len(self.acc))

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
                FIV = np.append(FIV, self.dt*integrate.trapz(ugf_pc))

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
    
    def get_amplitude_ims(self):
        
        """
        Get amplitude-based intensity measures of a ground-motion record
        Parameters
        ----------
        None        
        Returns
        -------
        record.pga: peak ground acceleration
        record.pgv: peak ground velocity
        record.pgd: peak ground displacement
        """
        record = eqsig.AccSignal(self.acc, self.dt)
        record.generate_displacement_and_velocity_series()
        return record.pga, record.pgv, record.pgd
    

    def get_arias_intensity(self, start = 0.05, end = 0.95):
        
        record = eqsig.AccSignal(self.acc, self.dt)    
        
        # Get the 5% and 95% time stamps
        t_start, t_end  = eqsig.im.calc_sig_dur_vals(self.acc, self.dt, start=start, end=end, se=True)
        # Get the index of the steps
        step_start = int(t_start/self.dt)
        step_end   = int(t_end/self.dt)
        
        # Get the arias intensity time-history
        AI = eqsig.im.calc_arias_intensity(record)
        # Get the arias intensity between the percentiles requested       
        
        return AI[step_end]-AI[step_start]


    def get_duration_ims(self):
        """
        Get duration-based intensity measures of a ground-motion record
        Parameters
        ----------
        None        
        Returns
        -------
        record.arias_intensity: arias intensity (AI)
        record.cav: cumulative absolute velocity (CAV)
        record.t_959: 5%-95% significant duration
        
        """        
        record = eqsig.AccSignal(self.acc, self.dt)
        record.generate_cumulative_stats()
        record.generate_duration_stats()
        return record.arias_intensity, record.cav, record.t_595

    def get_significant_duration(self, start = 0.05, end = 0.95):
        """
        Get significant duration of a ground-motion record
        Parameters
        ----------
        None        
        Returns
        -------
        record.arias_intensity: arias intensity (AI)
        record.cav: cumulative absolute velocity (CAV)
        record.t_959: 5%-95% significant duration
        
        """        
        return eqsig.im.calc_sig_dur_vals(self.acc, self.dt, start = start, end = end, se=False)
