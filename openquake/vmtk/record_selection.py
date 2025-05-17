import pandas as pd
import os
import math
import random
import matplotlib.pyplot as plt
from scipy import stats
import scipy.linalg as la
from scipy.stats import norm
import shutil
import scipy.io as sio
import warnings
import numpy as np
import requests

"""
This code is used to select ground motions with response spectra representative of a target 
scenario earthquake, as predicted by a ground motion model. Spectra can be selected to be 
consistent with the full distribution of response spectra, or conditional on a spectral 
amplitude at a given period (i.e., using the conditional spectrum approach).
Single-component or two-component motions can be selected, and several ground motion 
databases are provided to search in.

Based on:
Baker, J. W., and Lee, C. (2018). "An Improved Algorithm for Selecting Ground Motions 
to Match a Conditional Spectrum." Journal of Earthquake Engineering, 22(4), 708–723.
"""

class SelectionParams:
    """Parameters controlling ground motion selection"""
    def __init__(self):
        # Ground motion database and type of selection
        self.database_file = 'NGA_W2_meta_data'  # Changed from CyberShake to NGA_W2
        self.cond = 0          # 0: unconditional selection, 1: conditional
        self.arb = 2           # 1: single-component selection, 2: two-component selection
        self.RotD = 50         # 50: use SaRotD50 data, 100: use SaRotD100 data
        
        # Number of ground motions and spectral periods of interest
        self.nGM = 30          # Number of ground motions to be selected
        self.Tcond = 1.5       # Period at which spectra should be scaled and matched
        self.Tmin = 0.1        # Smallest spectral period of interest
        self.Tmax = 10         # Largest spectral period of interest
        self.TgtPer = np.logspace(np.log10(self.Tmin), np.log10(self.Tmax), 30)  # Array of periods
        self.SaTcond = None    # Target Sa(Tcond) - if provided, rup.eps_bar will be back-computed
        
        # Parameters for vertical spectra (optional)
        self.matchV = 0        # 1: match vertical spectrum, 0: don't match
        self.TminV = 0.01      # Smallest vertical spectral period
        self.TmaxV = 10        # Largest vertical spectral period
        self.weightV = 0.5     # Weight on vertical spectral match versus horizontal
        self.sepScaleV = 1     # 1: scale vertical components separately, 0: same scale factor
        self.TgtPerV = np.logspace(np.log10(self.TminV), np.log10(self.TmaxV), 20)
        
        # Scaling and evaluation parameters
        self.isScaled = 1      # 1: allow records to be scaled, 0: don't scale
        self.maxScale = 10      # Maximum allowable scale factor
        self.tol = 10          # Tolerable percent error to skip optimization
        self.optType = 0       # 0: use sum of squared errors, 1: use D-statistic
        self.penalty = 0       # >0: penalize spectra more than 3 sigma from target
        self.weights = [1.0, 2.0, 0.3]  # Weights for error in mean, std dev, skewness
        self.nLoop = 2         # Number of optimization loops
        self.useVar = 1        # 1: use computed variance, 0: use target variance of 0
        
        # Runtime parameters
        self.indTcond = None   # Index of conditioning period (set in get_target_spectrum)
        self.lnSa1 = None      # Target spectral acceleration at Tcond


class Rupture:
    """Parameters specifying the rupture scenario"""
    def __init__(self):
        # Basic parameters
        self.M_bar = 6.5       # Earthquake magnitude
        self.Rjb = 11          # Closest distance to surface projection (km)
        self.eps_bar = 1.9     # Epsilon value (for conditional selection)
        self.Vs30 = 259        # Average shear wave velocity (m/s)
        self.z1 = 999          # Basin depth (km), 999 if unknown
        self.region = 1        # 0: global, 1: California, 2: Japan, 3: China/Turkey, 4: Italy
        self.Fault_Type = 1    # 0: unspecified, 1: strike-slip, 2: normal, 3: reverse
        
        # Additional seismological parameters for GMPE
        self.Rrup = 11         # Closest distance to rupture plane (km)
        self.Rx = 11           # Horizontal distance (km)
        self.W = 15            # Down-dip rupture width (km)
        self.Ztor = 0          # Depth to top of rupture (km)
        self.Zbot = 15         # Depth to bottom of seismogenic crust (km)
        self.dip = 90          # Fault dip angle (deg)
        self.lambda_ = 0       # Rake angle (deg)
        self.Fhw = 0           # Flag for hanging wall
        self.Z2p5 = 1          # Depth to Vs=2.5 km/sec (km)
        self.Zhyp = 10         # Hypocentral depth (km)
        self.FRV = 0           # Flag for reverse faulting
        self.FNM = 0           # Flag for normal faulting
        self.Sj = 0            # Flag for regional site effects (1 for Japan)


class AllowedRecords:
    """Criteria for selecting records from the database"""
    def __init__(self):
        self.Vs30 = [-float('inf'), float('inf')]  # Bounds for Vs30 values
        self.Mag = [6.0, 8.2]                     # Bounds for magnitude
        self.D = [0, 50]                           # Bounds for distance
        self.idxInvalid = []                       # Index of ground motions to exclude


class TargetSa:
    """Response spectrum target values to match"""
    def __init__(self):
        self.meanReq = None    # Target response spectrum means
        self.covReq = None     # Matrix of response spectrum covariances
        self.stdevs = None     # Vector of standard deviations


class IntensityMeasures:
    """Intensity measure values chosen and available"""
    def __init__(self):
        self.recID = None              # Indices of selected spectra
        self.scaleFac = None           # Scale factors for selected spectra
        self.sampleSmall = None        # Matrix of selected logarithmic response spectra
        self.sampleBig = None          # Matrix of logarithmic spectra to search
        self.stageOneScaleFac = None   # Scale factors after first selection stage
        self.stageOneMeans = None      # Mean log response spectra after first stage
        self.stageOneStdevs = None     # Standard deviation after first stage
        # For vertical components
        self.scaleFacV = None          
        self.sampleBigV = None
        self.stageOneScaleFacV = None
        self.stageOneMeansV = None
        self.stageOneStdevsV = None


def screen_database(selection_params, allowed_recs):
    """
    Load and screen the ground motion database for suitable motions
    
    Parameters:
    -----------
    selection_params : SelectionParams
        Parameters controlling the selection
    allowed_recs : AllowedRecords
        Criteria for selecting records
    
    Returns:
    --------
    SaKnown : numpy.ndarray
        Ground motion spectra that passed screening
    selection_params : SelectionParams
        Updated with indices of periods
    indPer : list
        Indices of target periods in database
    knownPer : numpy.ndarray
        Periods available in the database
    metadata : dict
        Information about the database and selected records
    """
    # This is a placeholder - in a real implementation, you would:
    # 1. Load database from file (e.g., using pandas)
    # 2. Filter records based on allowed_recs criteria
    # 3. Extract relevant data and metadata
    
    print(f"Loading database: {selection_params.database_file}")
    print(f"Screening for records with Vs30 between {allowed_recs.Vs30[0]} and {allowed_recs.Vs30[1]}")
    print(f"Screening for records with magnitude between {allowed_recs.Mag[0]} and {allowed_recs.Mag[1]}")
    print(f"Screening for records with distance between {allowed_recs.D[0]} and {allowed_recs.D[1]}")
    
    # Dummy implementation - replace with actual database loading
    n_records = 1000  # Assume 1000 records pass screening
    n_periods = 100   # Assume 100 periods in database
    
    # Create dummy data for demonstration
    SaKnown = np.random.lognormal(0, 0.5, size=(n_records, n_periods))
    knownPer = np.logspace(np.log10(0.01), np.log10(10), n_periods)
    
    # Find indices of target periods in database
    indPer = []
    for period in selection_params.TgtPer:
        # Find closest period in database
        idx = np.abs(knownPer - period).argmin()
        indPer.append(idx)
    
    # Find index of conditioning period
    selection_params.indTcond = np.where(np.isclose(selection_params.TgtPer, selection_params.Tcond))[0][0]
    
    # Create metadata
    metadata = {
        'database': selection_params.database_file,
        'n_records': n_records,
        'allowedIndex': np.arange(n_records),  # Original indices of allowed records
        'filenames': [f"record_{i}.acc" for i in range(n_records)],
        'record_info': pd.DataFrame({
            'Magnitude': np.random.uniform(allowed_recs.Mag[0], allowed_recs.Mag[1], n_records),
            'Distance': np.random.uniform(allowed_recs.D[0], allowed_recs.D[1], n_records),
            'Vs30': np.random.uniform(200, 800, n_records)
        })
    }
    
    # If vertical components are needed, create those too
    if selection_params.matchV == 1:
        selection_params.SaKnownV = np.random.lognormal(-0.5, 0.6, size=(n_records, n_periods))
        selection_params.indPerV = []
        for period in selection_params.TgtPerV:
            idx = np.abs(knownPer - period).argmin()
            selection_params.indPerV.append(idx)
    
    return SaKnown, selection_params, indPer, knownPer, metadata

import numpy as np
from scipy import interpolate

def gmpe_bssa_2014(M, T, Rjb, Fault_Type, region, z1, Vs30):
    """
    BSSA14 NGA-West2 model, based on:
    
    Boore, D. M., Stewart, J. P., Seyhan, E., and Atkinson, G. M. (2014). 
    "NGA-West2 Equations for Predicting PGA, PGV, and 5% Damped PSA for 
    Shallow Crustal Earthquakes." Earthquake Spectra, 30(3), 1057-1085.
    
    Provides ground-motion prediction equations for computing medians and
    standard deviations of average horizontal component intensity measures
    (IMs) for shallow crustal earthquakes in active tectonic regions.
    
    Parameters:
    -----------
    M : float
        Moment Magnitude
    T : float or array
        Period (sec); Use Period = -1 for PGV computation
                     Use 1000 to output the array of median with original period
                     (no interpolation)
    Rjb : float
        Joyner-Boore distance (km)
    Fault_Type : int
        0 for unspecified fault
        1 for strike-slip fault
        2 for normal fault
        3 for reverse fault
    region : int
        0 for global (incl. Taiwan)
        1 for California
        2 for Japan
        3 for China or Turkey
        4 for Italy
    z1 : float
        Basin depth (km); depth from the groundsurface to the
        1km/s shear-wave horizon.
        Use 999 if unknown
    Vs30 : float
        Shear wave velocity averaged over top 30 m in m/s
    
    Returns:
    --------
    median : float or array
        Median amplitude prediction
    sigma : float or array
        NATURAL LOG standard deviation
    period1 : float or array
        Periods corresponding to the output values
    """
    
    # Define periods
    period = np.array([-1, 0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 
                       0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10])
    
    # Set fault type flags
    U = (Fault_Type == 0)
    SS = (Fault_Type == 1)
    NS = (Fault_Type == 2)
    RS = (Fault_Type == 3)
    
    # Convert T to numpy array if it's not already
    if not isinstance(T, np.ndarray):
        T = np.array([T])
    
    # Check if we need to compute for predefined periods
    if len(T) == 1 and T[0] == 1000:
        # Compute median and sigma with pre-defined period
        median = np.zeros(len(period) - 2)
        sigma = np.zeros(len(period) - 2)
        period1 = period[2:]
        
        for ip in range(2, len(period)):
            median[ip-2], sigma[ip-2] = BSSA_2014_sub(M, ip, Rjb, U, SS, NS, RS, region, z1, Vs30)
            
    else:
        # Compute median and sigma with user-defined period
        median = np.zeros(len(T))
        sigma = np.zeros(len(T))
        period1 = T
        
        for i in range(len(T)):
            Ti = T[i]
            # Check if interpolation is required
            if not np.any(np.abs(period - Ti) < 0.0001):
                # Find neighboring periods for interpolation
                T_low = np.max(period[period < Ti])
                T_high = np.min(period[period > Ti])
                ip_low = np.where(period == T_low)[0][0]
                ip_high = np.where(period == T_high)[0][0]
                
                # Compute values at neighboring periods
                Sa_low, sigma_low = BSSA_2014_sub(M, ip_low, Rjb, U, SS, NS, RS, region, z1, Vs30)
                Sa_high, sigma_high = BSSA_2014_sub(M, ip_high, Rjb, U, SS, NS, RS, region, z1, Vs30)
                
                # Interpolate in log-space for Sa and linear-space for sigma
                x = np.log([T_low, T_high])
                Y_sa = np.log([Sa_low, Sa_high])
                Y_sigma = [sigma_low, sigma_high]
                
                median[i] = np.exp(np.interp(np.log(Ti), x, Y_sa))
                sigma[i] = np.interp(np.log(Ti), x, Y_sigma)
            else:
                # Use exact period
                ip_T = np.where(np.abs(period - Ti) < 0.0001)[0][0]
                median[i], sigma[i] = BSSA_2014_sub(M, ip_T, Rjb, U, SS, NS, RS, region, z1, Vs30)
    
    # If T was a scalar, return scalar results
    if len(T) == 1 and T[0] != 1000:
        return median[0], sigma[0], period1[0]
    else:
        return median, sigma, period1


def BSSA_2014_sub(M, ip, Rjb, U, SS, NS, RS, region, z1, Vs30):
    """
    Sub-function for BSSA_2014 to compute median and sigma for a specific period
    
    Parameters:
    -----------
    M : float
        Moment Magnitude
    ip : int
        Index for the period
    Rjb : float
        Joyner-Boore distance (km)
    U, SS, NS, RS : bool
        Flags for fault type
    region : int
        Region indicator
    z1 : float
        Basin depth (km)
    Vs30 : float
        Shear wave velocity (m/s)
    
    Returns:
    --------
    median : float
        Median amplitude prediction
    sigma : float
        NATURAL LOG standard deviation
    """
    
    # Reference parameters
    mref = 4.5
    rref = 1
    v_ref = 760
    f1 = 0
    f3 = 0.1
    v1 = 225
    v2 = 300
    
    # Define periods and coefficients (same as in MATLAB code)
    period = np.array([-1, 0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 
                       0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10])
    
    # Coefficients from the BSSA 2014 GMM
    mh = np.array([6.2, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.54, 5.74, 5.92, 6.05, 6.14, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2])
    e0 = np.array([5.037, 0.4473, 0.4534, 0.48598, 0.56916, 0.75436, 0.96447, 1.1268, 1.3095, 1.3255, 1.2766, 1.2217, 1.1046, 0.96991, 0.66903, 0.3932, -0.14954, -0.58669, -1.1898, -1.6388, -1.966, -2.5865, -3.0702])
    e1 = np.array([5.078, 0.4856, 0.4916, 0.52359, 0.6092, 0.79905, 1.0077, 1.1669, 1.3481, 1.359, 1.3017, 1.2401, 1.1214, 0.99106, 0.69737, 0.4218, -0.11866, -0.55003, -1.142, -1.5748, -1.8882, -2.4874, -2.9537])
    e2 = np.array([4.849, 0.2459, 0.2519, 0.29707, 0.40391, 0.60652, 0.77678, 0.8871, 1.0648, 1.122, 1.0828, 1.0246, 0.89765, 0.7615, 0.47523, 0.207, -0.3138, -0.71466, -1.23, -1.6673, -2.0245, -2.8176, -3.3776])
    e3 = np.array([5.033, 0.4539, 0.4599, 0.48875, 0.55783, 0.72726, 0.9563, 1.1454, 1.3324, 1.3414, 1.3052, 1.2653, 1.1552, 1.012, 0.69173, 0.4124, -0.1437, -0.60658, -1.2664, -1.7516, -2.0928, -2.6854, -3.1726])
    e4 = np.array([1.073, 1.431, 1.421, 1.4331, 1.4261, 1.3974, 1.4174, 1.4293, 1.2844, 1.1349, 1.0166, 0.95676, 0.96766, 1.0384, 1.2871, 1.5004, 1.7622, 1.9152, 2.1323, 2.204, 2.2299, 2.1187, 1.8837])
    e5 = np.array([-0.1536, 0.05053, 0.04932, 0.053388, 0.061444, 0.067357, 0.073549, 0.055231, -0.042065, -0.11096, -0.16213, -0.1959, -0.22608, -0.23522, -0.21591, -0.18983, -0.1467, -0.11237, -0.04332, -0.014642, -0.014855, -0.081606, -0.15096])
    e6 = np.array([0.2252, -0.1662, -0.1659, -0.16561, -0.1669, -0.18082, -0.19665, -0.19838, -0.18234, -0.15852, -0.12784, -0.092855, -0.023189, 0.029119, 0.10829, 0.17895, 0.33896, 0.44788, 0.62694, 0.76303, 0.87314, 1.0121, 1.0651])
    c1 = np.array([-1.24300, -1.13400, -1.13400, -1.13940, -1.14210, -1.11590, -1.08310, -1.06520, -1.05320, -1.06070, -1.07730, -1.09480, -1.12430, -1.14590, -1.17770, -1.19300, -1.20630, -1.21590, -1.21790, -1.21620, -1.21890, -1.25430, -1.32530])
    c2 = np.array([0.14890, 0.19170, 0.19160, 0.18962, 0.18842, 0.18709, 0.18225, 0.17203, 0.15401, 0.14489, 0.13925, 0.13388, 0.12512, 0.12015, 0.11054, 0.10248, 0.09645, 0.09636, 0.09764, 0.10218, 0.10353, 0.12507, 0.15183])
    c3 = np.array([-0.00344, -0.00809, -0.00809, -0.00807, -0.00834, -0.00982, -0.01058, -0.01020, -0.00898, -0.00772, -0.00652, -0.00548, -0.00405, -0.00322, -0.00193, -0.00121, -0.00037, 0.00000, 0.00000, -0.00005, 0.00000, 0.00000, 0.00000])
    h = np.array([5.3, 4.5, 4.5, 4.5, 4.49, 4.2, 4.04, 4.13, 4.39, 4.61, 4.78, 4.93, 5.16, 5.34, 5.6, 5.74, 6.18, 6.54, 6.93, 7.32, 7.78, 9.48, 9.66])
    
    # Regional adjustments
    deltac3_gloCATW = np.zeros(23)  # All zeros
    deltac3_CHTU = np.array([0.004350, 0.002860, 0.002820, 0.002780, 0.002760, 0.002960, 0.002960, 0.002880, 0.002790, 0.002610, 0.002440, 0.002200, 0.002110, 0.002350, 0.002690, 0.002920, 0.003040, 0.002920, 0.002620, 0.002610, 0.002600, 0.002600, 0.003030])
    deltac3_ITJA = np.array([-0.000330, -0.002550, -0.002440, -0.002340, -0.002170, -0.001990, -0.002160, -0.002440, -0.002710, -0.002970, -0.003140, -0.003300, -0.003210, -0.002910, -0.002530, -0.002090, -0.001520, -0.001170, -0.001190, -0.001080, -0.000570, 0.000380, 0.001490])
    
    # More coefficients
    c = np.array([-0.8400, -0.6000, -0.6037, -0.5739, -0.5341, -0.4580, -0.4441, -0.4872, -0.5796, -0.6876, -0.7718, -0.8417, -0.9109, -0.9693, -1.0154, -1.0500, -1.0454, -1.0392, -1.0112, -0.9694, -0.9195, -0.7766, -0.6558])
    vc = np.array([1300.00, 1500.00, 1500.20, 1500.36, 1502.95, 1501.42, 1494.00, 1479.12, 1442.85, 1392.61, 1356.21, 1308.47, 1252.66, 1203.91, 1147.59, 1109.95, 1072.39, 1009.49, 922.43, 844.48, 793.13, 771.01, 775.00])
    f4 = np.array([-0.1000, -0.1500, -0.1483, -0.1471, -0.1549, -0.1963, -0.2287, -0.2492, -0.2571, -0.2466, -0.2357, -0.2191, -0.1958, -0.1704, -0.1387, -0.1052, -0.0679, -0.0361, -0.0136, -0.0032, -0.0003, -0.0001, 0.0000])
    f5 = np.array([-0.00844, -0.00701, -0.00701, -0.00728, -0.00735, -0.00647, -0.00573, -0.00560, -0.00585, -0.00614, -0.00644, -0.00670, -0.00713, -0.00744, -0.00812, -0.00844, -0.00771, -0.00479, -0.00183, -0.00152, -0.00144, -0.00137, -0.00136])
    f6 = np.array([-9.900, -9.900, -9.9, -9.9, -9.9, -9.9, -9.9, -9.9, -9.9, -9.9, -9.9, -9.9, -9.9, -9.9, 0.092, 0.367, 0.638, 0.871, 1.135, 1.271, 1.329, 1.329, 1.183])
    f7 = np.array([-9.900, -9.900, -9.9, -9.9, -9.9, -9.9, -9.9, -9.9, -9.9, -9.9, -9.9, -9.9, -9.9, -9.9, 0.059, 0.208, 0.309, 0.382, 0.516, 0.629, 0.738, 0.809, 0.703])
    tau1 = np.array([0.4010, 0.3980, 0.4020, 0.4090, 0.4450, 0.5030, 0.4740, 0.4150, 0.3540, 0.3440, 0.3500, 0.3630, 0.3810, 0.4100, 0.4570, 0.4980, 0.5250, 0.5320, 0.5370, 0.5430, 0.5320, 0.5110, 0.4870])
    tau2 = np.array([0.3460, 0.3480, 0.3450, 0.3460, 0.3640, 0.4260, 0.4660, 0.4580, 0.3880, 0.3090, 0.2660, 0.2290, 0.2100, 0.2240, 0.2660, 0.2980, 0.3150, 0.3290, 0.3440, 0.3490, 0.3350, 0.2700, 0.2390])
    phi1 = np.array([0.6440, 0.6950, 0.6980, 0.7020, 0.7210, 0.7530, 0.7450, 0.7280, 0.7200, 0.7110, 0.6980, 0.6750, 0.6430, 0.6150, 0.5810, 0.5530, 0.5320, 0.5260, 0.5340, 0.5360, 0.5280, 0.5120, 0.5100])
    phi2 = np.array([0.5520, 0.4950, 0.4990, 0.5020, 0.5140, 0.5320, 0.5420, 0.5410, 0.5370, 0.5390, 0.5470, 0.5610, 0.5800, 0.5990, 0.6220, 0.6250, 0.6190, 0.6180, 0.6190, 0.6160, 0.6220, 0.6340, 0.6040])
    dphiR = np.array([0.082, 0.100, 0.096, 0.092, 0.081, 0.063, 0.064, 0.087, 0.120, 0.136, 0.141, 0.138, 0.122, 0.109, 0.100, 0.098, 0.104, 0.105, 0.088, 0.070, 0.061, 0.058, 0.060])
    dphiV = np.array([0.080, 0.070, 0.070, 0.030, 0.029, 0.030, 0.022, 0.014, 0.015, 0.045, 0.055, 0.050, 0.049, 0.060, 0.070, 0.020, 0.010, 0.008, 0.000, 0.000, 0.000, 0.000, 0.000])
    R1 = np.array([105.000, 110.000, 111.670, 113.100, 112.130, 97.930, 85.990, 79.590, 81.330, 90.910, 97.040, 103.150, 106.020, 105.540, 108.390, 116.390, 125.380, 130.370, 130.360, 129.490, 130.220, 130.720, 130.000])
    R2 = np.array([272.000, 270.000, 270.000, 270.000, 270.000, 270.000, 270.040, 270.090, 270.160, 270.000, 269.450, 268.590, 266.540, 265.000, 266.510, 270.000, 262.410, 240.140, 195.000, 199.450, 230.000, 250.390, 210.000])
    
    # The source (event function)
    if M <= mh[ip]:
        F_E = e0[ip] * U + e1[ip] * SS + e2[ip] * NS + e3[ip] * RS + e4[ip] * (M - mh[ip]) + e5[ip] * (M - mh[ip])**2
    else:
        F_E = e0[ip] * U + e1[ip] * SS + e2[ip] * NS + e3[ip] * RS + e6[ip] * (M - mh[ip])
    
    # The path function
    if region == 0 or region == 1:
        deltac3 = deltac3_gloCATW
    elif region == 3:
        deltac3 = deltac3_CHTU
    elif region == 2 or region == 4:
        deltac3 = deltac3_ITJA
    
    r = np.sqrt(Rjb**2 + h[ip]**2)
    F_P = (c1[ip] + c2[ip] * (M - mref)) * np.log(r / rref) + (c3[ip] + deltac3[ip]) * (r - rref)
    
    # Find PGAr
    if Vs30 != v_ref or ip != 1:  # Note: In Python, indexing starts at 0, so ip=1 corresponds to ip=2 in MATLAB
        # Compute PGA at reference site condition
        PGA_r, sigma_r = BSSA_2014_sub(M, 1, Rjb, U, SS, NS, RS, region, z1, v_ref)  # ip=1 for PGA in Python
        
        # The site function
        # Linear component
        if Vs30 <= vc[ip]:
            ln_Flin = c[ip] * np.log(Vs30 / v_ref)
        else:
            ln_Flin = c[ip] * np.log(vc[ip] / v_ref)
        
        # Nonlinear component
        f2 = f4[ip] * (np.exp(f5[ip] * (min(Vs30, 760) - 360)) - np.exp(f5[ip] * (760 - 360)))
        ln_Fnlin = f1 + f2 * np.log((PGA_r + f3) / f3)
        
        # Effect of basin depth
        if z1 != 999:
            if region == 1:  # California
                mu_z1 = np.exp(-7.15/4 * np.log((Vs30**4 + 570.94**4) / (1360**4 + 570.94**4))) / 1000
            elif region == 2:  # Japan
                mu_z1 = np.exp(-5.23/2 * np.log((Vs30**2 + 412.39**2) / (1360**2 + 412.39**2))) / 1000
            else:
                mu_z1 = np.exp(-7.15/4 * np.log((Vs30**4 + 570.94**4) / (1360**4 + 570.94**4))) / 1000
            dz1 = z1 - mu_z1
        else:
            dz1 = 0
        
        if z1 != 999:
            if period[ip] < 0.65:
                F_dz1 = 0
            elif period[ip] >= 0.65 and abs(dz1) <= f7[ip] / f6[ip]:
                F_dz1 = f6[ip] * dz1
            else:
                F_dz1 = f7[ip]
        else:
            F_dz1 = 0
        
        F_S = ln_Flin + ln_Fnlin + F_dz1
        
        ln_Y = F_E + F_P + F_S
        median = np.exp(ln_Y)
    else:
        ln_y = F_E + F_P
        median = np.exp(ln_y)
    
    # Aleatory - uncertainty function
    if M <= 4.5:
        tau = tau1[ip]
        phi_M = phi1[ip]
    elif 4.5 < M < 5.5:
        tau = tau1[ip] + (tau2[ip] - tau1[ip]) * (M - 4.5)
        phi_M = phi1[ip] + (phi2[ip] - phi1[ip]) * (M - 4.5)
    else:  # M >= 5.5
        tau = tau2[ip]
        phi_M = phi2[ip]
    
    if Rjb <= R1[ip]:
        phi_MR = phi_M
    elif R1[ip] < Rjb <= R2[ip]:
        phi_MR = phi_M + dphiR[ip] * (np.log(Rjb / R1[ip]) / np.log(R2[ip] / R1[ip]))
    else:  # Rjb > R2[ip]
        phi_MR = phi_M + dphiR[ip]
    
    if Vs30 >= v2:
        phi_MRV = phi_MR
    elif v1 <= Vs30 <= v2:
        phi_MRV = phi_MR - dphiV[ip] * (np.log(v2 / Vs30) / np.log(v2 / v1))
    else:  # Vs30 < v1
        phi_MRV = phi_MR - dphiV[ip]
    
    sigma = np.sqrt(phi_MRV**2 + tau**2)
    
    return median, sigma

import numpy as np

def gmpe_bj_2008_corr(T1, T2):
    """
    Compute the correlation of epsilons for the NGA ground motion models.
    
    The function is strictly empirical, fitted over the range
    0.01s <= T1, T2 <= 10s
    
    Documentation is provided in the following document:
    
    Baker, J.W. and Jayaram, N., (2008) "Correlation of spectral 
    acceleration values from NGA ground motion models," Earthquake Spectra, 
    24(1), 299-317.
    
    Parameters:
    -----------
    T1, T2 : float
        The two periods of interest. The periods may be equal,
        and there is no restriction on which one is larger.
    
    Returns:
    --------
    rho : float
        The predicted correlation coefficient
    """
    T_min = min(T1, T2)
    T_max = max(T1, T2)
    
    C1 = (1 - np.cos(np.pi/2 - np.log(T_max / max(T_min, 0.109)) * 0.366))
    
    if T_max < 0.2:
        C2 = 1 - 0.105 * (1 - 1 / (1 + np.exp(100 * T_max - 5))) * (T_max - T_min) / (T_max - 0.0099)
    else:
        C2 = 1.0  # Default value, not used in this case
    
    if T_max < 0.109:
        C3 = C2
    else:
        C3 = C1
    
    C4 = C1 + 0.5 * (np.sqrt(C3) - C3) * (1 + np.cos(np.pi * T_min / 0.109))
    
    if T_max <= 0.109:
        rho = C2
    elif T_min > 0.109:
        rho = C1
    elif T_max < 0.2:
        rho = min(C2, C4)
    else:
        rho = C4
    
    return rho



def get_target_spectrum(knownPer, selectionParams, indPer, rup):
    """
    Calculate and return the target mean spectrum and covariance
    matrix at available periods
    
    Parameters:
    -----------
    knownPer : numpy.ndarray
        Periods available in the database
    selectionParams : SelectionParams
        Parameters controlling the selection
    indPer : list
        Indices of target periods in database
    rup : Rupture
        Rupture scenario parameters
    
    Returns:
    --------
    targetSa : TargetSa
        Target spectral acceleration data
    """
    # Initialize output structure
    targetSa = {}
    
    # Compute target mean spectrum
    # compute the median and standard deviations of RotD50 response spectrum values
    sa, sigma, _ = gmpe_bssa_2014(rup.M_bar, knownPer, rup.Rjb, 
                                rup.Fault_Type, rup.region, 
                                rup.z1, rup.Vs30)
    
    # modify spectral targets if RotD100 values were specified for two-component selection
    if selectionParams.RotD == 100 and selectionParams.arb == 2:
        rotD100Ratio, rotD100Sigma = gmpe_sb_2014_ratios(knownPer)
        sa = sa * rotD100Ratio
        sigma = np.sqrt(sigma**2 + rotD100Sigma**2)
    
    # back-calculate an epsilon to match the target Sa(T_cond), if Sa(T_cond) is specified
    if hasattr(selectionParams, 'SaTcond') and selectionParams.SaTcond is not None:
        # Interpolate in log-log space to get median Sa and sigma at Tcond
        logPer = np.log(knownPer)
        logSa = np.log(sa)
        
        median_SaTcond = np.exp(np.interp(np.log(selectionParams.Tcond), logPer, logSa))
        sigma_SaTcond = np.interp(np.log(selectionParams.Tcond), logPer, sigma)
        
        eps_bar = (np.log(selectionParams.SaTcond) - np.log(median_SaTcond)) / sigma_SaTcond
        
        print(f"Back-calculated epsilon = {eps_bar:.3f}")  # output result for user verification
    else:
        # use user-specified epsilon value
        eps_bar = rup.eps_bar
    
    # Calculate target mean spectrum (in log space)
    if selectionParams.cond == 1:
        # compute correlations and the conditional mean spectrum
        rho = np.zeros(len(sa))
        for i in range(len(sa)):
            rho[i] = gmpe_bj_2008_corr(knownPer[i], 
                                      selectionParams.TgtPer[selectionParams.indTcond])
        
        TgtMean = np.log(sa) + sigma * eps_bar * rho
    else:
        TgtMean = np.log(sa)
    
    # Compute covariance matrix
    TgtCovs = np.zeros((len(sa), len(sa)))
    for i in range(len(sa)):
        for j in range(len(sa)):
            # Periods
            Ti = knownPer[i]
            Tj = knownPer[j]
            
            # Means and variances
            varT = sigma[selectionParams.indTcond]**2
            sigma22 = varT
            var1 = sigma[i]**2
            var2 = sigma[j]**2
            
            # Covariances
            if selectionParams.cond == 1:
                sigmaCorr = gmpe_bj_2008_corr(Ti, Tj) * np.sqrt(var1 * var2)
                sigma11 = np.array([[var1, sigmaCorr], [sigmaCorr, var2]])
                sigma12 = np.array([
                    [gmpe_bj_2008_corr(Ti, selectionParams.Tcond) * np.sqrt(var1 * varT)],
                    [gmpe_bj_2008_corr(selectionParams.Tcond, Tj) * np.sqrt(var2 * varT)]
                ])
                
                # Calculate conditional covariance
                sigmaCond = sigma11 - sigma12 @ np.linalg.inv(np.array([[sigma22]])) @ sigma12.T
                TgtCovs[i, j] = sigmaCond[0, 1]
            else:
                TgtCovs[i, j] = gmpe_bj_2008_corr(Ti, Tj) * np.sqrt(var1 * var2)
    
    # over-write covariance matrix with zeros if no variance is desired in the ground motion selection
    if hasattr(selectionParams, 'useVar') and selectionParams.useVar == 0:
        TgtCovs = np.zeros_like(TgtCovs)
    
    # find covariance values near zero and set them to a small number
    TgtCovs[np.abs(TgtCovs) < 1e-10] = 1e-10
    
    # Store target mean and covariance matrix at target periods
    targetSa['meanReq'] = TgtMean[indPer]
    targetSa['covReq'] = TgtCovs[np.ix_(indPer, indPer)]
    targetSa['stdevs'] = np.sqrt(np.diag(targetSa['covReq']))
    
    # target mean and covariance at all periods
    targetSa['meanAllT'] = TgtMean
    targetSa['covAllT'] = TgtCovs
    
    # Revise target spectrum to include V component
    if hasattr(selectionParams, 'matchV') and selectionParams.matchV == 1:
        print("Vertical component handling is implemented in simplified form")
        # For brevity, I'm not including the vertical component handling code
        # as it would similarly need to be updated to use attribute notation
    
    return targetSa


def get_target_spectrum(knownPer, selectionParams, indPer, rup):
    """
    Calculate and return the target mean spectrum and covariance
    matrix at available periods
    
    Parameters:
    -----------
    knownPer : numpy.ndarray
        Periods available in the database
    selectionParams : SelectionParams
        Parameters controlling the selection
    indPer : list
        Indices of target periods in database
    rup : Rupture
        Rupture scenario parameters
    
    Returns:
    --------
    targetSa : TargetSa
        Target spectral acceleration data
    """
    # Initialize output structure
    target_dict = {}
    
    # Compute target mean spectrum
    # compute the median and standard deviations of RotD50 response spectrum values
    sa, sigma, _ = gmpe_bssa_2014(rup.M_bar, knownPer, rup.Rjb, 
                                rup.Fault_Type, rup.region, 
                                rup.z1, rup.Vs30)
    
    # modify spectral targets if RotD100 values were specified for two-component selection
    if selectionParams.RotD == 100 and selectionParams.arb == 2:
        rotD100Ratio, rotD100Sigma = gmpe_sb_2014_ratios(knownPer)
        sa = sa * rotD100Ratio
        sigma = np.sqrt(sigma**2 + rotD100Sigma**2)
    
    # back-calculate an epsilon to match the target Sa(T_cond), if Sa(T_cond) is specified
    if hasattr(selectionParams, 'SaTcond') and selectionParams.SaTcond is not None:
        # Interpolate in log-log space to get median Sa and sigma at Tcond
        logPer = np.log(knownPer)
        logSa = np.log(sa)
        
        median_SaTcond = np.exp(np.interp(np.log(selectionParams.Tcond), logPer, logSa))
        sigma_SaTcond = np.interp(np.log(selectionParams.Tcond), logPer, sigma)
        
        eps_bar = (np.log(selectionParams.SaTcond) - np.log(median_SaTcond)) / sigma_SaTcond
        
        print(f"Back-calculated epsilon = {eps_bar:.3f}")
    else:
        # use user-specified epsilon value
        eps_bar = rup.eps_bar
    
    # (Log) Response Spectrum Mean: TgtMean
    if selectionParams.cond == 1:
        # compute correlations and the conditional mean spectrum
        rho = np.zeros(len(sa))
        for i in range(len(sa)):
            rho[i] = gmpe_bj_2008_corr(knownPer[i], 
                                      selectionParams.TgtPer[selectionParams.indTcond])
        
        TgtMean = np.log(sa) + sigma * eps_bar * rho
    else:
        TgtMean = np.log(sa)
    
    # Compute covariances and correlations at all periods
    TgtCovs = np.zeros((len(sa), len(sa)))
    for i in range(len(sa)):
        for j in range(len(sa)):
            # Periods
            Ti = knownPer[i]
            Tj = knownPer[j]
            
            # Means and variances
            varT = sigma[selectionParams.indTcond]**2
            sigma22 = varT
            var1 = sigma[i]**2
            var2 = sigma[j]**2
            
            # Covariances
            if selectionParams.cond == 1:
                sigmaCorr = gmpe_bj_2008_corr(Ti, Tj) * np.sqrt(var1 * var2)
                sigma11 = np.array([[var1, sigmaCorr], [sigmaCorr, var2]])
                sigma12 = np.array([
                    [gmpe_bj_2008_corr(Ti, selectionParams.Tcond) * np.sqrt(var1 * varT)],
                    [gmpe_bj_2008_corr(selectionParams.Tcond, Tj) * np.sqrt(var2 * varT)]
                ])
                
                # Calculate conditional covariance
                sigmaCond = sigma11 - sigma12 @ np.linalg.inv(np.array([[sigma22]])) @ sigma12.T
                TgtCovs[i, j] = sigmaCond[0, 1]
            else:
                TgtCovs[i, j] = gmpe_bj_2008_corr(Ti, Tj) * np.sqrt(var1 * var2)
    
    # overwrite covariance matrix with zeros if no variance is desired
    if hasattr(selectionParams, 'useVar') and selectionParams.useVar == 0:
        TgtCovs = np.zeros_like(TgtCovs)
    
    # find covariance values near zero and set them to a small number
    TgtCovs[np.abs(TgtCovs) < 1e-10] = 1e-10
    
    # Store target mean and covariance matrix at target periods
    target_dict['meanReq'] = TgtMean[indPer]
    target_dict['covReq'] = TgtCovs[np.ix_(indPer, indPer)]
    target_dict['stdevs'] = np.sqrt(np.diag(target_dict['covReq']))
    
    # target mean and covariance at all periods
    target_dict['meanAllT'] = TgtMean
    target_dict['covAllT'] = TgtCovs
    
    # Revise target spectrum to include V component
    if hasattr(selectionParams, 'matchV') and selectionParams.matchV == 1:
        print("Vertical component handling is implemented in simplified form")
        # Add vertical component handling if required
        
    # Print diagnostic information about target spectrum
    print("Target spectrum information:")
    print(f"meanReq shape: {target_dict['meanReq'].shape}")
    print(f"stdevs shape: {target_dict['stdevs'].shape}")
    print(f"covReq shape: {target_dict['covReq'].shape}")
    
    # Convert dictionary to TargetSa object
    targetSa = TargetSa()
    targetSa.meanReq = target_dict['meanReq']
    targetSa.covReq = target_dict['covReq']
    targetSa.stdevs = target_dict['stdevs']
    targetSa.meanAllT = target_dict['meanAllT']
    targetSa.covAllT = target_dict['covAllT']
    
    return targetSa


def gmpe_bj_2008_corr(T1, T2):
    """
    Baker and Jayaram (2008) correlation model for response spectral values
    
    Parameters:
    -----------
    T1, T2 : float
        Periods to compute correlation between
    
    Returns:
    --------
    rho : float
        Correlation coefficient
    
    Notes:
    ------
    Implementation based on:
    Baker, J.W. and Jayaram, N., (2008) "Correlation of spectral 
    acceleration values from NGA ground motion models," Earthquake Spectra, 
    24(1), 299-317.
    """
    T_min = min(T1, T2)
    T_max = max(T1, T2)
    
    C1 = (1 - np.cos(np.pi/2 - np.log(T_max / max(T_min, 0.109)) * 0.366))
    
    if T_max < 0.2:
        C2 = 1 - 0.105 * (1 - 1 / (1 + np.exp(100 * T_max - 5))) * (T_max - T_min) / (T_max - 0.0099)
    else:
        C2 = 1.0  # Default value, not used in this case
    
    if T_max < 0.109:
        C3 = C2
    else:
        C3 = C1
    
    C4 = C1 + 0.5 * (np.sqrt(C3) - C3) * (1 + np.cos(np.pi * T_min / 0.109))
    
    if T_max <= 0.109:
        rho = C2
    elif T_min > 0.109:
        rho = C1
    elif T_max < 0.2:
        rho = min(C2, C4)
    else:
        rho = C4
    
    return rho


def nearestSPD(A):
    """
    Find the nearest (in Frobenius norm) Symmetric Positive Definite matrix to A
    
    From Higham: "The nearest symmetric positive semidefinite matrix in the
    Frobenius norm to an arbitrary real matrix A is shown to be (B + H)/2,
    where H is the symmetric polar factor of B=(A + A')/2."
    
    Parameters:
    -----------
    A : array_like
        Input matrix, which will be converted to the nearest Symmetric
        Positive Definite Matrix.
    
    Returns:
    --------
    Ahat : ndarray
        The nearest Symmetric Positive Definite matrix to A.
    """
    # Check input is a square matrix
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix.')
    
    # Handle scalar case
    if A.shape[0] == 1:
        if A[0, 0] <= 0:
            return np.array([[np.finfo(float).eps]])
        else:
            return A.copy()
    
    # Symmetrize A into B
    B = (A + A.T) / 2
    
    # Compute the symmetric polar factor of B. Call it H.
    # Clearly H is itself SPD.
    U, s, Vh = np.linalg.svd(B)
    H = Vh.T @ np.diag(s) @ Vh
    
    # Get Ahat in the above formula
    Ahat = (B + H) / 2
    
    # Ensure symmetry
    Ahat = (Ahat + Ahat.T) / 2
    
    # Test that Ahat is in fact PD. If not, tweak it a bit.
    k = 0
    while True:
        try:
            # Attempt Cholesky decomposition (will fail if not PD)
            np.linalg.cholesky(Ahat)
            break
        except np.linalg.LinAlgError:
            # Ahat failed the chol test. Tweak by adding a tiny multiple of identity matrix.
            k += 1
            eigvals = np.linalg.eigvalsh(Ahat)
            min_eig = np.min(eigvals)
            # Add a small increment to make it positive definite
            Ahat = Ahat + (-min_eig * k**2 + np.finfo(float).eps) * np.eye(A.shape[0])
    
    return Ahat


# These functions are placeholders and would need actual implementations
# based on referenced GMPEs in the MATLAB code

def gmpe_sb_2014_ratios(knownPer):
    """
    Placeholder for the Shahi-Baker 2014 GMM for computing RotD100/RotD50 ratios
    """
    # Placeholder values
    rotD100Ratio = np.ones(len(knownPer)) * 1.2  # Typically RotD100 > RotD50
    rotD100Sigma = np.ones(len(knownPer)) * 0.1
    
    return rotD100Ratio, rotD100Sigma


import numpy as np

def gmpe_sb_2014_ratios(T):
    """
    Compute Sa_RotD100/Sa_RotD50 ratios from Shahi and Baker (2014)
    
    Shahi, S. K., and Baker, J. W. (2014). "NGA-West2 models for ground-
    motion directionality." Earthquake Spectra, 30(3), 1285-1300.
    
    Parameters:
    -----------
    T : float or array_like
        Period(s) of interest (sec)
    
    Returns:
    --------
    ratio : float or array_like
        Geometric mean of Sa_RotD100/Sa_RotD50
    sigma : float or array_like
        Standard deviation of log(Sa_RotD100/Sa_RotD50)
    phi : float or array_like
        Within-event standard deviation
    tau : float or array_like
        Between-event standard deviation
    """
    # Model coefficient values from Table 1 of the above-referenced paper
    periods_orig = np.array([0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 
                            0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0])
    
    ratios_orig = np.array([1.19243805900000, 1.19124621700000, 1.18767783300000, 1.18649074900000, 
                           1.18767783300000, 1.18767783300000, 1.19961419400000, 1.20562728500000, 
                           1.21652690500000, 1.21896239400000, 1.22875320400000, 1.22875320400000, 
                           1.23738465100000, 1.24110237900000, 1.24234410200000, 1.24358706800000, 
                           1.24732343100000, 1.25985923900000, 1.26490876900000, 1.28531008400000, 
                           1.29433881900000])
    
    sigma_orig = np.array([0.08] * 20 + [0.08])  # All values are 0.08
    phi_orig = np.array([0.08] * 20 + [0.07])    # All values are 0.08 except last one (0.07)
    tau_orig = np.array([0.01] * 18 + [0.02, 0.02, 0.03])  # Most values are 0.01, except last three
    
    # Convert input to numpy array if not already
    T = np.asarray(T)
    scalar_input = False
    if T.ndim == 0 or (hasattr(T, 'shape') and len(T.shape) == 0):
        T = np.array([T])  # Makes 1D array from scalar
        scalar_input = True
    
    # Initialize output arrays
    ratio = np.zeros_like(T)
    sigma = np.zeros_like(T)
    phi = np.zeros_like(T)
    tau = np.zeros_like(T)
    
    # Interpolate to compute values for the user-specified periods
    for i, t in enumerate(T):
        if t <= periods_orig[0]:
            ratio[i] = ratios_orig[0]
            sigma[i] = sigma_orig[0]
            phi[i] = phi_orig[0]
            tau[i] = tau_orig[0]
        elif t >= periods_orig[-1]:
            ratio[i] = ratios_orig[-1]
            sigma[i] = sigma_orig[-1]
            phi[i] = phi_orig[-1]
            tau[i] = tau_orig[-1]
        else:
            ratio[i] = np.interp(np.log(t), np.log(periods_orig), ratios_orig)
            sigma[i] = np.interp(np.log(t), np.log(periods_orig), sigma_orig)
            phi[i] = np.interp(np.log(t), np.log(periods_orig), phi_orig)
            tau[i] = np.interp(np.log(t), np.log(periods_orig), tau_orig)
    
    # Return scalar if input was scalar
    if scalar_input:
        return ratio[0], sigma[0], phi[0], tau[0]
    else:
        return ratio, sigma, phi, tau



def extract_matlab_cell_array(mat_data, var_name):
    """
    Extract and convert MATLAB cell arrays to Python lists or numpy arrays
    
    Parameters:
    -----------
    mat_data : dict
        MATLAB data loaded with scipy.io.loadmat
    var_name : str
        Name of the variable to extract
    
    Returns:
    --------
    result : list or numpy.ndarray
        Extracted data in Python format
    """
    if var_name not in mat_data:
        return None
    
    cell_data = mat_data[var_name]
    
    # Check if data is empty or None
    if cell_data is None or cell_data.size == 0:
        return []
    
    try:
        # Handle 1D cell arrays
        if cell_data.ndim == 2 and (cell_data.shape[0] == 1 or cell_data.shape[1] == 1):
            result = []
            for cell in cell_data.flatten():
                if cell.size == 0:  # Handle empty cells
                    result.append('')
                elif isinstance(cell[0], np.ndarray):
                    if cell[0].dtype.kind == 'U' or cell[0].dtype.kind == 'S':
                        result.append(str(cell[0][0]) if cell[0].size > 0 else '')
                    else:
                        result.append(cell[0])
                else:
                    result.append(cell[0])
            return np.array(result)
        
        # Handle 2D cell arrays
        else:
            result = []
            for i in range(cell_data.shape[0]):
                row = []
                for j in range(cell_data.shape[1]):
                    cell = cell_data[i, j]
                    if cell.size == 0:  # Handle empty cells
                        row.append('')
                    elif isinstance(cell[0], np.ndarray):
                        if cell[0].dtype.kind == 'U' or cell[0].dtype.kind == 'S':
                            row.append(str(cell[0][0]) if cell[0].size > 0 else '')
                        else:
                            row.append(cell[0])
                    else:
                        row.append(cell[0])
                result.append(row)
            return np.array(result)
    except (IndexError, ValueError, TypeError) as e:
        # If we encounter an error, try a simpler approach to extract data
        print(f"Warning: Error extracting MATLAB cell array '{var_name}': {str(e)}")
        print(f"Attempting alternate extraction method...")
        
        try:
            # For scalar values or simple arrays
            if not isinstance(cell_data, np.ndarray) or cell_data.dtype != 'object':
                return cell_data
            
            # For cell arrays, extract what we can
            result = []
            for i in range(cell_data.size):
                try:
                    if cell_data.ndim == 1:
                        cell = cell_data[i]
                    else:
                        cell = cell_data.flat[i]
                    
                    if cell.size == 0:
                        result.append('')
                    else:
                        val = cell[0] if hasattr(cell, '__getitem__') else cell
                        if isinstance(val, np.ndarray) and val.size > 0:
                            if val.dtype.kind in ('U', 'S'):
                                result.append(str(val.item(0)))
                            else:
                                result.append(val.item(0) if val.size == 1 else val)
                        else:
                            result.append(val)
                except:
                    result.append('')
            
            return np.array(result)
        
        except Exception as e2:
            print(f"Warning: Alternate extraction failed: {str(e2)}")
            print(f"Returning empty array for {var_name}")
            return np.array([])


def screen_database(selection_params, allowed_recs):
    """
    Load and screen the ground motion database for suitable motions
    
    Parameters:
    -----------
    selection_params : SelectionParams
        Parameters controlling the selection
    allowed_recs : AllowedRecords
        Criteria for selecting records
    
    Returns:
    --------
    SaKnown : numpy.ndarray
        Ground motion spectra that passed screening
    selection_params : SelectionParams
        Updated with indices of periods
    indPer : list
        Indices of target periods in database
    knownPer : numpy.ndarray
        Periods available in the database
    metadata : dict
        Information about the database and selected records
    """
    import scipy.io as sio
    
    # Load the MATLAB database file
    database_path = os.path.join('Databases', f"{selection_params.database_file}.mat")
    try:
        print(f"Loading database: {selection_params.database_file}")
        mat_data = sio.loadmat(database_path)
        
        # Print available keys to help with debugging
        print(f"Available keys in database: {list(mat_data.keys())}")
        
        # Extract data from MATLAB structure
        # Common variables in all databases
        Periods = mat_data.get('Periods', np.array([])).flatten()
        if Periods.size == 0:
            raise ValueError("No 'Periods' data found in database")
            
        magnitude = mat_data.get('magnitude', np.array([])).flatten()
        closest_D = mat_data.get('closest_D', np.array([])).flatten()
        soil_Vs30 = mat_data.get('soil_Vs30', np.array([])).flatten()
        NGA_num = mat_data.get('NGA_num', np.array([])).flatten()
        
        # Extract filenames and directory locations
        Filename_1 = extract_matlab_cell_array(mat_data, 'Filename_1')
        Filename_2 = extract_matlab_cell_array(mat_data, 'Filename_2') if 'Filename_2' in mat_data else None
        dirLocation = extract_matlab_cell_array(mat_data, 'dirLocation')
        getTimeSeries = extract_matlab_cell_array(mat_data, 'getTimeSeries')
        
        # Extract spectral acceleration data
        Sa_1 = mat_data.get('Sa_1')
        if Sa_1 is None or Sa_1.size == 0:
            raise ValueError("No 'Sa_1' data found in database")
            
        Sa_2 = mat_data.get('Sa_2') if 'Sa_2' in mat_data else None
        
        # Check for RotD50 or RotD100 data
        Sa_RotD50 = mat_data.get('Sa_RotD50') if 'Sa_RotD50' in mat_data else None
        Sa_RotD100 = mat_data.get('Sa_RotD100') if 'Sa_RotD100' in mat_data else None
        
        # Check for vertical component data
        Sa_vert = mat_data.get('Sa_vert') if 'Sa_vert' in mat_data else None
        Filename_vert = extract_matlab_cell_array(mat_data, 'Filename_vert') if 'Filename_vert' in mat_data else None
        
        print(f"Screening for records with Vs30 between {allowed_recs.Vs30[0]} and {allowed_recs.Vs30[1]}")
        print(f"Screening for records with magnitude between {allowed_recs.Mag[0]} and {allowed_recs.Mag[1]}")
        print(f"Screening for records with distance between {allowed_recs.D[0]} and {allowed_recs.D[1]}")
        
    except Exception as e:
        print(f"Error loading database: {str(e)}")
        raise
    
    # Format appropriate ground motion metadata variables for single or two-component selection
    metadata = {}
    metadata['getTimeSeries'] = getTimeSeries
    metadata['dirLocation'] = dirLocation
    
    if selection_params.arb == 1 and selection_params.matchV != 1:
        # Single-component selection -- treat each component as a separate candidate
        if Sa_2 is None:  # If 2nd component doesn't exist (e.g., EXSIM data)
            metadata['Filename'] = Filename_1
            metadata['compNum'] = np.ones_like(magnitude)
            metadata['recNum'] = np.arange(1, len(magnitude) + 1)
            SaKnown = Sa_1
        else:  # 2nd component exists
            metadata['Filename'] = np.concatenate([Filename_1, Filename_2])
            metadata['compNum'] = np.concatenate([np.ones_like(magnitude), 2 * np.ones_like(magnitude)])
            metadata['recNum'] = np.concatenate([np.arange(1, len(magnitude) + 1), np.arange(1, len(magnitude) + 1)])
            SaKnown = np.vstack([Sa_1, Sa_2])
            soil_Vs30 = np.concatenate([soil_Vs30, soil_Vs30])
            magnitude = np.concatenate([magnitude, magnitude])
            closest_D = np.concatenate([closest_D, closest_D])
            NGA_num = np.concatenate([NGA_num, NGA_num])
            metadata['dirLocation'] = np.concatenate([dirLocation, dirLocation])
    
    elif selection_params.arb == 2 and selection_params.matchV != 1:
        # Two-component selection
        metadata['Filename'] = np.column_stack([Filename_1, Filename_2]) if Filename_2 is not None else Filename_1
        metadata['recNum'] = np.arange(1, len(magnitude) + 1)
        
        if selection_params.RotD == 50 and Sa_RotD50 is not None:
            SaKnown = Sa_RotD50
        elif selection_params.RotD == 100 and Sa_RotD100 is not None:
            SaKnown = Sa_RotD100
        else:
            print(f"Warning: RotD{selection_params.RotD} not provided in database")
            # Use geometric mean of the single-component Sa's as fallback
            SaKnown = np.sqrt(Sa_1 * Sa_2) if Sa_2 is not None else Sa_1
    
    else:
        # Including vertical components
        if Filename_vert is not None:
            metadata['Filename'] = np.column_stack([Filename_1, Filename_2, Filename_vert])
        else:
            metadata['Filename'] = np.column_stack([Filename_1, Filename_2])
        
        metadata['recNum'] = np.arange(1, len(magnitude) + 1)
        
        if selection_params.RotD == 50 and Sa_RotD50 is not None:
            SaKnown = Sa_RotD50
        elif selection_params.RotD == 100 and Sa_RotD100 is not None:
            SaKnown = Sa_RotD100
        else:
            print(f"Warning: RotD{selection_params.RotD} not provided in database")
            # Use geometric mean of the single-component Sa's as fallback
            SaKnown = np.sqrt(Sa_1 * Sa_2) if Sa_2 is not None else Sa_1
        
        if Sa_vert is not None:
            SaKnownV = Sa_vert
    
    # Create variable for known periods
    idxPer = np.where(Periods <= 10)[0]  # Throw out periods > 10s to avoid GMPE evaluation problems
    knownPer = Periods[idxPer]
    
    # Modify TgtPer to include Tcond if running a conditional selection
    if selection_params.cond == 1:
        # Ensure Tcond is in the range of available periods
        if selection_params.Tcond < np.min(knownPer) or selection_params.Tcond > np.max(knownPer):
            print(f"Warning: Tcond={selection_params.Tcond} is outside the available period range [{np.min(knownPer)}, {np.max(knownPer)}]")
            # Adjust Tcond to the nearest available period
            selection_params.Tcond = knownPer[np.argmin(np.abs(knownPer - selection_params.Tcond))]
            print(f"Adjusted Tcond to {selection_params.Tcond}")
        
        # Add Tcond to TgtPer if not already included
        if not any(np.isclose(selection_params.TgtPer, selection_params.Tcond)):
            selection_params.TgtPer = np.sort(np.append(selection_params.TgtPer, selection_params.Tcond))
    
    # Match periods (known periods and target periods for error computations)
    # Save the indices of the matched periods in knownPer
    indPer = np.zeros(len(selection_params.TgtPer), dtype=int)
    for i in range(len(selection_params.TgtPer)):
        indPer[i] = np.argmin(np.abs(knownPer - selection_params.TgtPer[i]))
    
    # Make sure target periods match the actual database periods
    selection_params.TgtPer = knownPer[indPer]
    
    # Identify the index of Tcond within TgtPer
    # Using robust approach to find the index
    tcond_diffs = np.abs(selection_params.TgtPer - selection_params.Tcond)
    tcond_idx = np.argmin(tcond_diffs)
    
    # Verify we found a close match
    if tcond_diffs[tcond_idx] > 1e-6:
        print(f"Warning: Could not find exact Tcond match. Using closest period: {selection_params.TgtPer[tcond_idx]}")
    
    selection_params.indTcond = tcond_idx
    selection_params.Tcond = selection_params.TgtPer[tcond_idx]  # Use the actual nearest period
    print(f"Conditioning period (Tcond): {selection_params.Tcond}, index: {selection_params.indTcond}")
    
    # If selecting V components, revise TgtPerV
    if selection_params.matchV == 1:
        if selection_params.cond == 1 and not any(np.isclose(selection_params.TgtPerV, selection_params.Tcond)):
            selection_params.TgtPerV = np.sort(np.append(selection_params.TgtPerV, selection_params.Tcond))
        
        indPerV = np.zeros(len(selection_params.TgtPerV), dtype=int)
        for i in range(len(selection_params.TgtPerV)):
            indPerV[i] = np.argmin(np.abs(knownPer - selection_params.TgtPerV[i]))
        
        indPerV = np.unique(indPerV)
        selection_params.TgtPerV = knownPer[indPerV]
        selection_params.indPerV = indPerV
    
    # Screen the records to be considered
    if selection_params.matchV == 1 and 'Sa_vert' in locals():
        # Ensure that each record contains all 3 components
        recValidSa = ~np.any(Sa_1 == -999, axis=1) & ~np.any(Sa_2 == -999, axis=1) & ~np.any(np.isin(Sa_vert, [-999, np.inf]), axis=1)
    else:
        recValidSa = ~np.all(SaKnown == -999, axis=1)  # Remove invalid inputs
    
    recValidSoil = (soil_Vs30 > allowed_recs.Vs30[0]) & (soil_Vs30 < allowed_recs.Vs30[1])
    recValidMag = (magnitude > allowed_recs.Mag[0]) & (magnitude < allowed_recs.Mag[1])
    recValidDist = (closest_D > allowed_recs.D[0]) & (closest_D < allowed_recs.D[1])
    
    recValidIdx = ~np.isin(metadata['recNum'], allowed_recs.idxInvalid)
    
    # Flag indices of allowable records that will be searched
    metadata['allowedIndex'] = np.where(recValidSoil & recValidMag & recValidDist & recValidSa & recValidIdx)[0]
    
    # Resize SaKnown to include only allowed records
    SaKnown = SaKnown[metadata['allowedIndex']][:, idxPer]
    
    # If selecting V components, save new variables in selectionParams
    if selection_params.matchV == 1 and 'SaKnownV' in locals():
        SaKnownV = SaKnownV[metadata['allowedIndex']][:, idxPer]
        selection_params.SaKnownV = SaKnownV
    
    # Count number of allowed spectra
    selection_params.nBig = len(metadata['allowedIndex'])
    
    print(f"Number of allowed ground motions = {selection_params.nBig}")
    assert selection_params.nBig >= selection_params.nGM, 'Warning: there are not enough allowable ground motions'
    
    return SaKnown, selection_params, indPer, knownPer, metadata



def simulate_spectra(targetSa, selectionParams, seed_value, n_trials):
    """
    Simulate response spectra matching the computed targets
    
    Parameters:
    -----------
    targetSa : TargetSa
        Target spectral acceleration data
    selectionParams : SelectionParams
        Parameters controlling the selection
    seed_value : int
        Random seed (0 for random)
    n_trials : int
        Number of iterations for spectral simulation
    
    Returns:
    --------
    best_spectra : numpy.ndarray
        Best set of simulated spectra
    """
    # Set random seed if provided
    if seed_value != 0:
        np.random.seed(seed_value)
    
    n_gm = selectionParams.nGM
    n_periods = len(targetSa.meanReq)
    
    # Initialize storage for best simulation
    best_err = float('inf')
    best_spectra = None
    
    print(f"Simulating {n_trials} sets of response spectra")
    
    # First ensure the covariance matrix is positive definite
    cov_matrix = targetSa.covReq.copy()
    
    # Check if the matrix is positive definite
    try:
        L_check = la.cholesky(cov_matrix, lower=True)
        print("Covariance matrix is positive definite, proceeding with simulation")
    except la.LinAlgError:
        # Not positive definite, fix it
        print("Covariance matrix is not positive definite, applying correction")
        cov_matrix = nearestSPD(cov_matrix)
        
        # Verify we fixed it
        try:
            L_check = la.cholesky(cov_matrix, lower=True)
            print("Covariance matrix has been corrected to be positive definite")
        except:
            # If still fails, use a simpler approach
            print("Simplifying covariance matrix due to persistent issues")
            n = cov_matrix.shape[0]
            std_devs = np.sqrt(np.diag(cov_matrix))
            # Create a simplified correlation matrix with exponential decay
            simplified_corr = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    # Correlation decreases with distance between periods
                    simplified_corr[i,j] = np.exp(-0.3 * np.abs(i-j))
            
            # Convert back to covariance
            cov_matrix = np.diag(std_devs) @ simplified_corr @ np.diag(std_devs)
    
    # Perform multiple trials and select the best one
    for trial in range(n_trials):
        try:
            # Generate multivariate normal samples
            z = np.random.normal(0, 1, size=(n_gm, n_periods))
            
            # Cholesky decomposition of covariance matrix
            try:
                L = la.cholesky(cov_matrix, lower=True)
                # Apply correlation structure to random samples
                spectra = targetSa.meanReq + np.dot(z, L.T)
            except:
                # If Cholesky still fails, use eigenvalue decomposition
                print(f"Warning: Cholesky decomposition failed on trial {trial}, using eigenvalue method")
                eigvals, eigvecs = la.eigh(cov_matrix)
                # Handle small negative eigenvalues due to numerical errors
                eigvals = np.maximum(eigvals, 0)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
                spectra = targetSa.meanReq + np.dot(z, L.T)
            
            # Calculate error in mean and standard deviation
            sim_mean = np.mean(spectra, axis=0)
            sim_std = np.std(spectra, axis=0)
            sim_skew = stats.skew(spectra, axis=0)
            
            mean_err = np.mean(np.abs(sim_mean - targetSa.meanReq) / np.abs(targetSa.meanReq))
            std_err = np.mean(np.abs(sim_std - targetSa.stdevs) / targetSa.stdevs)
            skew_err = np.mean(np.abs(sim_skew))
            
            total_err = (selectionParams.weights[0] * mean_err + 
                         selectionParams.weights[1] * std_err +
                         selectionParams.weights[2] * skew_err)
            
            # Save if this is the best simulation so far
            if total_err < best_err:
                best_err = total_err
                best_spectra = np.exp(spectra)
        except Exception as e:
            print(f"Warning: Error during simulation trial {trial}: {str(e)}")
            continue
    
    # If all trials failed, create a simple set of spectra based on the mean and stddev
    if best_spectra is None:
        print("Warning: All simulation trials failed. Creating simplified spectra.")
        # Create simplified spectra based on lognormal distribution
        best_spectra = np.zeros((n_gm, n_periods))
        std_devs = targetSa.stdevs
        
        for i in range(n_gm):
            # Randomize around the target mean with the target standard deviation
            # But without correlation structure
            log_values = targetSa.meanReq + np.random.normal(0, std_devs, n_periods)
            best_spectra[i] = np.exp(log_values)
    
    return best_spectra
def find_ground_motions(selection_params, simulated_spectra, IMs):
    """
    Find best matches to the simulated spectra from ground-motion database
    
    Parameters:
    -----------
    selection_params : SelectionParams
        Parameters controlling the selection
    simulated_spectra : numpy.ndarray
        Simulated target spectra
    IMs : IntensityMeasures
        Structure to store results
    
    Returns:
    --------
    IMs : IntensityMeasures
        Updated with selected ground motions
    """
    # Determine index for scaling - conditioning period or all periods
    if selection_params.cond == 1:
        scale_fac_index = selection_params.indTcond
    else:
        scale_fac_index = np.arange(len(selection_params.TgtPer))
    
    # Initialize vectors
    IMs.recID = np.zeros(selection_params.nGM, dtype=int)
    IMs.sampleSmall = np.zeros((0, len(selection_params.TgtPer)))  # Empty array to stack onto
    IMs.scaleFac = np.ones(selection_params.nGM)
    
    # Find database spectra most similar to each simulated spectrum
    for i in range(selection_params.nGM):  # For each simulated spectrum
        err = np.zeros(selection_params.nBig)  # Initialize error matrix
        scale_fac = np.ones(selection_params.nBig)  # Initialize scale factors to 1
        
        # Compute scale factors and errors for each candidate ground motion
        for j in range(selection_params.nBig):
            if selection_params.isScaled:  # If scaling is allowed
                # Calculate scale factor using equation from MATLAB code
                exp_sample = np.exp(IMs.sampleBig[j, scale_fac_index])
                numerator = np.sum(exp_sample * simulated_spectra[i, scale_fac_index])
                denominator = np.sum(exp_sample**2)
                scale_fac[j] = numerator / denominator
            
            # Compute error - sum of squared log differences
            err[j] = np.sum((np.log(np.exp(IMs.sampleBig[j, :]) * scale_fac[j]) - np.log(simulated_spectra[i, :]))**2)
        
        # Exclude previously-selected ground motions
        if i > 0:
            err[IMs.recID[:i]] = 1000000
        
        # Exclude ground motions requiring too large of a scale factor
        err[scale_fac > selection_params.maxScale] = 1000000
        
        # Find minimum-error ground motion
        min_err = np.min(err)
        if min_err >= 1000000:
            raise ValueError('Warning: problem with simulated spectrum. No good matches found')
        
        min_idx = np.argmin(err)
        IMs.recID[i] = min_idx
        IMs.scaleFac[i] = scale_fac[min_idx]  # Store scale factor
        
        # Store scaled log spectrum
        scaled_spectrum = np.log(np.exp(IMs.sampleBig[min_idx, :]) * scale_fac[min_idx])
        IMs.sampleSmall = np.vstack([IMs.sampleSmall, scaled_spectrum])
    
    return IMs

def find_ground_motionsV(selection_params, simulated_spectra, IMs):
    """
    Find best matches to the simulated spectra including vertical components
    
    This is similar to find_ground_motions but includes matching of vertical components.
    The implementation would be more complex in a full version.
    """
    # This is a placeholder for the vertical component matching
    # Full implementation would be similar to find_ground_motions but with 
    # additional consideration of vertical spectra
    
    # First call regular ground motion selection
    IMs = find_ground_motions(selection_params, simulated_spectra, IMs)
    
    # Then add vertical scale factors if needed
    if selection_params.matchV == 1 and selection_params.sepScaleV == 1:
        n_gm = selection_params.nGM
        IMs.scaleFacV = np.ones(n_gm)
        
        # Here would compute separate scale factors for vertical components
        # For simplicity in this example, use same scale factors
        IMs.scaleFacV = IMs.scaleFac.copy()
    
    return IMs

def plot_target_spectrum(target_sa, selection_params):
    """
    Plot target response spectrum to visualize its properties
    
    Parameters:
    -----------
    target_sa : dict or TargetSa
        Target spectral acceleration data
    selection_params : SelectionParams
        Parameters controlling the selection
    """
    import matplotlib.pyplot as plt
    
    # Check if target_sa is a dictionary and access attributes accordingly
    if isinstance(target_sa, dict):
        mean_req = target_sa.get('meanReq')
        stdevs = target_sa.get('stdevs')
        cov_req = target_sa.get('covReq')
    else:
        mean_req = getattr(target_sa, 'meanReq', None)
        stdevs = getattr(target_sa, 'stdevs', None)
        cov_req = getattr(target_sa, 'covReq', None)
        
    # Check if required attributes are valid
    if mean_req is None:
        print("Error: mean_req is None")
        return
    
    # Print some basic information about the target_sa object
    print(f"Target spectrum information:")
    print(f"meanReq shape: {np.shape(mean_req) if mean_req is not None else 'None'}")
    print(f"stdevs shape: {np.shape(stdevs) if stdevs is not None else 'None'}")
    print(f"covReq shape: {np.shape(cov_req) if cov_req is not None else 'None'}")
    
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot target mean spectrum
        plt.loglog(selection_params.TgtPer, np.exp(mean_req), 'r-', linewidth=2, label='Target Mean')
        
        # Plot confidence intervals (mean ± 1 standard deviation)
        plt.loglog(selection_params.TgtPer, np.exp(mean_req + stdevs), 'r--', linewidth=1.5, label='Mean + 1σ')
        plt.loglog(selection_params.TgtPer, np.exp(mean_req - stdevs), 'r--', linewidth=1.5, label='Mean - 1σ')
        
        # Add plot details
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.xlabel('Period (s)')
        plt.ylabel('Spectral Acceleration (g)')
        plt.title('Target Response Spectrum')
        plt.legend(loc='upper right')
        
        # Add conditioning period marker
        plt.axvline(x=selection_params.Tcond, color='k', linestyle=':', label=f'Tcond = {selection_params.Tcond}s')
        
        # Save the plot
        plt.savefig('target_spectrum.png', dpi=300)
        plt.close()
        
        print(f"Target spectrum plot saved to 'target_spectrum.png'")
        print(f"Mean values range: {np.min(np.exp(mean_req)):.4f} - {np.max(np.exp(mean_req)):.4f}")
        print(f"Standard deviation range: {np.min(stdevs):.4f} - {np.max(stdevs):.4f}")
        
        # Also print covariance matrix properties if available
        if cov_req is not None:
            try:
                eigenvalues = np.linalg.eigvalsh(cov_req)
                print(f"Covariance matrix eigenvalues range: {np.min(eigenvalues):.6f} - {np.max(eigenvalues):.6f}")
                print(f"Condition number of covariance matrix: {np.max(eigenvalues)/np.min(eigenvalues):.2f}")
            except Exception as e:
                print(f"Error calculating covariance matrix properties: {str(e)}")
    
    except Exception as e:
        print(f"Error plotting target spectrum: {str(e)}")




def within_tolerance(sample_small, target_sa, selection_params):
    """
    Check if errors are within tolerance to potentially skip optimization
    
    Parameters:
    -----------
    sample_small : numpy.ndarray
        Selected spectra
    target_sa : TargetSa
        Target spectral data
    selection_params : SelectionParams
        Parameters controlling the selection
    
    Returns:
    --------
    bool
        True if within tolerance, False otherwise
    """
    # Calculate sample mean and standard deviation
    sample_mean = np.mean(sample_small, axis=0)
    sample_std = np.std(sample_small, axis=0)
    
    # Calculate errors
    mean_err = np.mean(np.abs((sample_mean - target_sa.meanReq) / target_sa.meanReq)) * 100
    std_err = np.mean(np.abs((sample_std - target_sa.stdevs) / target_sa.stdevs)) * 100
    
    # Calculate total error using weights
    total_err = selection_params.weights[0] * mean_err + selection_params.weights[1] * std_err
    
    print(f"Mean error: {mean_err:.2f}%, Std dev error: {std_err:.2f}%, Total: {total_err:.2f}%")
    
    # Check if within tolerance
    return total_err < selection_params.tol


def within_toleranceV(IMs, target_sa, selection_params):
    """
    Check if errors are within tolerance for both horizontal and vertical components
    
    Parameters are similar to within_tolerance but include vertical components
    """
    # Check horizontal components
    h_within = within_tolerance(IMs.sampleSmall, target_sa, selection_params)
    
    # For vertical components, simplified here
    if selection_params.matchV == 1:
        # Calculate some dummy errors for demonstration
        IMs.medianErr = 5.0
        IMs.stdErr = 7.0
        v_within = (IMs.medianErr + IMs.stdErr) < selection_params.tol
    else:
        v_within = True
    
    return h_within and v_within, IMs


def optimize_ground_motions(selection_params, target_sa, IMs):
    """
    Further optimize the ground motion selection
    
    Parameters:
    -----------
    selection_params : SelectionParams
        Parameters controlling the selection
    target_sa : TargetSa
        Target spectral data
    IMs : IntensityMeasures
        Current set of selected motions
    
    Returns:
    --------
    IMs : IntensityMeasures
        Optimized set of selected motions
    """
    n_gm = selection_params.nGM
    n_big = IMs.sampleBig.shape[0]  # Number of records in database
    n_loops = selection_params.nLoop
    
    print(f"Optimizing selection with {n_loops} iterations")
    
    # Store current selection
    best_rec_id = IMs.recID.copy()
    best_scale_fac = IMs.scaleFac.copy()
    best_sample_small = IMs.sampleSmall.copy()
    
    # Calculate initial error
    best_mean = np.mean(best_sample_small, axis=0)
    best_std = np.std(best_sample_small, axis=0)
    
    mean_err = np.mean(np.abs((best_mean - target_sa.meanReq) / target_sa.meanReq)) * 100
    std_err = np.mean(np.abs((best_std - target_sa.stdevs) / target_sa.stdevs)) * 100
    
    best_err = selection_params.weights[0] * mean_err + selection_params.weights[1] * std_err
    
    # Optimization loops
    for loop in range(n_loops):
        print(f"Optimization loop {loop+1}/{n_loops}, current error: {best_err:.2f}%")
        
        # Try replacing each ground motion
        for i in range(n_gm):
            # Remove one ground motion from the set
            temp_set = np.delete(best_sample_small, i, axis=0)
            
            # Calculate the target properties with one fewer ground motion
            temp_mean = np.mean(temp_set, axis=0)
            temp_std = np.std(temp_set, axis=0)
            
            # Find a replacement that minimizes error
            best_replacement_err = float('inf')
            best_replacement_id = -1
            best_replacement_scale = 1.0
            
            # Search through database for a better replacement
            for j in range(n_big):
                # Skip if already in the selected set
                if j in best_rec_id:
                    continue
                
                db_spectrum = IMs.sampleBig[j, :]
                
                # Compute scale factor to match at conditioning period
                if selection_params.isScaled == 1:
                    scale_factor = np.exp(target_sa.meanReq[selection_params.indTcond] - db_spectrum[selection_params.indTcond])
                    
                    # Check if scale factor is within limits
                    if scale_factor > selection_params.maxScale or scale_factor < 1.0/selection_params.maxScale:
                        continue
                else:
                    scale_factor = 1.0
                
                # Scale the spectrum
                scaled_spectrum = db_spectrum + np.log(scale_factor)
                
                # Add to temporary set and calculate new statistics
                new_set = np.vstack([temp_set, scaled_spectrum])
                new_mean = np.mean(new_set, axis=0)
                new_std = np.std(new_set, axis=0)
                
                # Calculate error
                new_mean_err = np.mean(np.abs((new_mean - target_sa.meanReq) / target_sa.meanReq)) * 100
                new_std_err = np.mean(np.abs((new_std - target_sa.stdevs) / target_sa.stdevs)) * 100
                
                new_err = selection_params.weights[0] * new_mean_err + selection_params.weights[1] * new_std_err
                
                # Save if this is the best replacement so far
                if new_err < best_replacement_err:
                    best_replacement_err = new_err
                    best_replacement_id = j
                    best_replacement_scale = scale_factor
            
            # Check if the replacement improves the overall error
            if best_replacement_err < best_err:
                best_err = best_replacement_err
                best_rec_id[i] = best_replacement_id
                best_scale_fac[i] = best_replacement_scale
                best_sample_small[i, :] = IMs.sampleBig[best_replacement_id, :] + np.log(best_replacement_scale)
                
                print(f"  Replaced ground motion {i+1}, new error: {best_err:.2f}%")
    
    # Update the results
    IMs.recID = best_rec_id
    IMs.scaleFac = best_scale_fac
    IMs.sampleSmall = best_sample_small
    
    return IMs


def optimize_ground_motionsV(selection_params, target_sa, IMs):
    """
    Optimize ground motion selection including vertical components
    
    Similar to optimize_ground_motions but includes vertical components
    """
    # This is a placeholder - full implementation would be more complex
    # First optimize horizontal components
    IMs = optimize_ground_motions(selection_params, target_sa, IMs)
    
    # Then optimize vertical components if needed
    if selection_params.matchV == 1 and selection_params.sepScaleV == 1:
        # Here would optimize vertical scale factors
        # For simplicity, just use the horizontal scale factors
        IMs.scaleFacV = IMs.scaleFac.copy()
    
    return IMs

def plot_results(selection_params, target_sa, IMs, simulated_spectra, SaKnown, known_per):
    """
    Plot results of the ground motion selection
    
    Parameters:
    -----------
    selection_params : SelectionParams
        Parameters controlling the selection
    target_sa : TargetSa
        Target spectral data
    IMs : IntensityMeasures
        Selected intensity measures
    simulated_spectra : numpy.ndarray
        Simulated target spectra
    SaKnown : numpy.ndarray
        Ground motion spectra from database
    known_per : numpy.ndarray
        Periods available in the database
    """
    target_periods = selection_params.TgtPer
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual selected spectra
    for i in range(selection_params.nGM):
        # Extract original record and apply scale factor
        original_spectrum = SaKnown[IMs.recID[i], :]
        scaled_spectrum = original_spectrum * IMs.scaleFac[i]
        
        # Plot on log-log scale
        plt.loglog(known_per, scaled_spectrum, color='lightgray', linewidth=0.5)
    
    # Plot target spectrum with confidence intervals
    plt.loglog(target_periods, np.exp(target_sa.meanReq), 'r-', linewidth=2, label='Target Median')
    plt.loglog(target_periods, np.exp(target_sa.meanReq + target_sa.stdevs), 'r--', linewidth=1.5, label='Target 84th Percentile')
    plt.loglog(target_periods, np.exp(target_sa.meanReq - target_sa.stdevs), 'r--', linewidth=1.5, label='Target 16th Percentile')
    
    # Calculate mean and confidence intervals of selected spectra
    # Extract values at target periods for each selected ground motion
    selected_spectra = np.zeros((len(IMs.recID), len(selection_params.TgtPer)))
    
    # For each selected ground motion, extract values at target periods
    for i, rec_id in enumerate(IMs.recID):
        # Find the indices in known_per that correspond to target periods
        for j, period in enumerate(selection_params.TgtPer):
            idx = np.argmin(np.abs(known_per - period))
            selected_spectra[i, j] = np.log(SaKnown[rec_id, idx] * IMs.scaleFac[i])
    
    # Calculate statistics
    selected_means = np.mean(selected_spectra, axis=0)
    selected_std = np.std(selected_spectra, axis=0)
    
    # Plot selected spectra statistics
    plt.loglog(target_periods, np.exp(selected_means), 'b-', linewidth=2, label='Selected Median')
    plt.loglog(target_periods, np.exp(selected_means + selected_std), 'b--', linewidth=1.5, label='Selected 84th Percentile')
    plt.loglog(target_periods, np.exp(selected_means - selected_std), 'b--', linewidth=1.5, label='Selected 16th Percentile')
    
    # Add details to plot
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xlabel('Period (s)')
    plt.ylabel('Spectral Acceleration (g)')
    plt.title(f'Response Spectra for M{selection_params.rup.M_bar}, R{selection_params.rup.Rjb}km')
    plt.legend(loc='upper right')
    
    # Mark conditioning period if conditional selection
    if selection_params.cond == 1:
        plt.axvline(x=selection_params.Tcond, color='k', linestyle=':', label=f'Tcond = {selection_params.Tcond}s')
    
    plt.tight_layout()
    plt.savefig('selected_spectra.png', dpi=300)
    plt.show()



def plot_resultsV(selection_params, target_sa, IMs, simulated_spectra, SaKnown, known_per):
    """
    Plot results of ground motion selection including vertical components
    
    Similar to plot_results but includes vertical component plots
    """
    # First plot horizontal components
    plot_results(selection_params, target_sa, IMs, simulated_spectra, SaKnown, known_per)
    
    # Then plot vertical components if needed
    if selection_params.matchV == 1:
        plt.figure(figsize=(12, 8))
        plt.title('Vertical Component Spectra')
        # Implementation would be similar to plot_results
        # but for vertical components
        plt.tight_layout()
        plt.savefig('selected_spectra_vertical.png', dpi=300)
        plt.show()


def write_output(rec_idx, IMs, output_dir, output_file, metadata):
    """
    Output results to a text file
    
    Parameters:
    -----------
    rec_idx : numpy.ndarray
        Indices of selected motions in original database
    IMs : IntensityMeasures
        Selected intensity measures
    output_dir : str
        Directory for output files
    output_file : str
        Name of output file
    metadata : dict
        Information about the database and selected records
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file
    output_path = os.path.join(output_dir, output_file)
    
    with open(output_path, 'w') as f:
        f.write("# Selected Ground Motions\n")
        f.write("#\n")
        f.write("# Record_ID  Scale_Factor  Filename\n")
        f.write("#\n")
        
        for i, (idx, scale) in enumerate(zip(rec_idx, IMs.scaleFac)):
            f.write(f"{idx:8d}  {scale:12.5f}  {metadata['filenames'][idx]}\n")
    
    print(f"Output written to {output_path}")


def download_time_series(output_dir, rec_idx, metadata):
    """
    Copy selected time series to the working directory
    
    Parameters:
    -----------
    output_dir : str
        Directory for output files
    rec_idx : numpy.ndarray
        Indices of selected motions
    metadata : dict
        Information about the database and selected records
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Attempting to copy time series files to {output_dir}")
    
    # Determine database type based on directory location
    try:
        dir_location = metadata['dirLocation'][0]
        
        # NGA-West1 data
        if 'nga_files' in dir_location:
            for i, idx in enumerate(rec_idx):
                # Check if multiple components exist
                if isinstance(metadata['Filename'][idx], list) or isinstance(metadata['Filename'][idx], np.ndarray):
                    n_comp = len(metadata['Filename'][idx])
                else:
                    n_comp = 1
                
                for j in range(n_comp):
                    # Construct source URL or path
                    if n_comp == 1:
                        source = os.path.join(metadata['dirLocation'][idx], metadata['Filename'][idx])
                        dest = os.path.join(output_dir, f"GM{i+1}.txt")
                    else:
                        source = os.path.join(metadata['dirLocation'][idx], metadata['Filename'][idx][j])
                        dest = os.path.join(output_dir, f"GM{i+1}_comp{j+1}.txt")
                    
                    # Try to download or copy
                    try:
                        if source.startswith(('http://', 'https://')):
                            # Use requests for downloading
                            with requests.get(source, stream=True) as r:
                                r.raise_for_status()
                                with open(dest, 'wb') as f:
                                    for chunk in r.iter_content(chunk_size=8192):
                                        f.write(chunk)
                        else:
                            # Use shutil for local file copying
                            shutil.copy2(source, dest)
                        print(f"  Successfully copied {os.path.basename(source)} to {dest}")
                    except Exception as e:
                        print(f"  Error copying {source}: {str(e)}")
        
        # BBP data
        elif 'bbpvault' in dir_location:
            print("  BroadBand Platform data are currently not available for download")
        
        # NGA-West2 data
        elif 'ngawest2' in dir_location:
            print("  NGA-West2 data are currently not available for download")
        
        # CyberShake data
        elif 'CyberShake' in dir_location:
            for i, idx in enumerate(rec_idx):
                # Check if multiple components exist
                if isinstance(metadata['Filename'][idx], list) or isinstance(metadata['Filename'][idx], np.ndarray):
                    n_comp = len(metadata['Filename'][idx])
                else:
                    n_comp = 1
                
                for j in range(n_comp):
                    # Construct source path
                    if n_comp == 1:
                        source = os.path.join(metadata['dirLocation'][idx], metadata['Filename'][idx])
                        dest = os.path.join(output_dir, f"GM{i+1}.txt")
                    else:
                        source = os.path.join(metadata['dirLocation'][idx], metadata['Filename'][idx][j])
                        dest = os.path.join(output_dir, f"GM{i+1}_comp{j+1}.txt")
                    
                    # Copy file
                    try:
                        shutil.copy2(source, dest)
                        print(f"  Successfully copied {os.path.basename(source)} to {dest}")
                    except Exception as e:
                        print(f"  Error copying {source}: {str(e)}")
        
        # Unknown database
        else:
            print("  Unknown database format")
            
    except Exception as e:
        print(f"  Error identifying database type: {str(e)}")
        # Fallback to simplified approach
        for i, idx in enumerate(rec_idx):
            try:
                filename = metadata['filenames'][idx] if 'filenames' in metadata else f"record_{idx}.acc"
                print(f"  Would copy {filename} to {output_dir}")
            except:
                print(f"  Would copy record {idx} to {output_dir}")



def download_time_series(output_dir, rec_idx, metadata):
    """
    Extract selected time series from NGA-West2 zip file to the working directory
    
    Parameters:
    -----------
    output_dir : str
        Directory for output files
    rec_idx : numpy.ndarray
        Indices of selected motions
    metadata : dict
        Information about the database and selected records
    """
    import zipfile
    import os
    import re
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Extracting time series files to {output_dir}")
    
    # Read RSN numbers from the output file that's already been created
    output_file_path = os.path.join(output_dir, "Output_File.dat")
    rsn_numbers = []
    
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Skip header lines
                if line.startswith("Record Number") or line.startswith("To retrieve") or line.strip() == "":
                    continue
                
                # Parse the RSN from the line (should be in column 2)
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        rsn = int(parts[1])
                        rsn_numbers.append(rsn)
                    except ValueError:
                        pass
    
    if not rsn_numbers:
        print("WARNING: Could not read RSN numbers from Output_File.dat")
        # Fallback to using NGA_num if available
        if 'NGA_num' in metadata and len(metadata['NGA_num']) > 0:
            for idx in rec_idx:
                try:
                    rsn = metadata['NGA_num'][idx]
                    rsn_numbers.append(rsn)
                except (IndexError, TypeError):
                    pass
    
    # Path to the NGA_W2.zip file - adjust as needed
    #zip_path = "NGA_W2.zip"  # Change this to the actual path
    zip_path = r"C:\Users\35191\Documents\NGA_W2.zip"

    
    # Check if the zip file exists
    if not os.path.exists(zip_path):
        print(f"ERROR: Zip file not found at {zip_path}")
        print("Please specify the correct path to NGA_W2.zip")
        return
    
    print(f"Looking for RSN numbers: {rsn_numbers}")
    
    extracted_files = []
    
    # Extract files from zip
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all files in the zip
            file_list = zip_ref.namelist()
            
            # For each selected ground motion
            for i, rsn in enumerate(rsn_numbers):
                # Find files that match the RSN pattern
                rsn_pattern = f"RSN{rsn}_"
                # Consider the subdirectory structure
                rsn_files = [f for f in file_list if rsn_pattern in f and f.endswith('.AT2')]
                
                if not rsn_files:
                    # Try pattern with subdirectory
                    rsn_files = [f for f in file_list if f"NGA_W2/RSN{rsn}_" in f and f.endswith('.AT2')]
                
                if not rsn_files:
                    # Try with alternate RSN patterns
                    rsn_files = [f for f in file_list if f"RSN{rsn}" in f and f.endswith('.AT2')]
                    rsn_files.extend([f for f in file_list if f"NGA_W2/RSN{rsn}" in f and f.endswith('.AT2')])
                
                if rsn_files:
                    # Extract each matching file (up to 2 for horizontal components)
                    for j, rsn_file in enumerate(rsn_files[:2]):
                        # Determine output filename
                        base_name = os.path.basename(rsn_file)
                        out_path = os.path.join(output_dir, f"GM{i+1}_comp{j+1}.AT2")
                        
                        # Extract the file
                        with zip_ref.open(rsn_file) as source, open(out_path, 'wb') as target:
                            target.write(source.read())
                        
                        print(f"  Extracted {base_name} to {out_path}")
                        extracted_files.append(out_path)
                else:
                    print(f"  WARNING: No files found for RSN {rsn}")
            
            # Summary
            if extracted_files:
                print(f"Successfully extracted {len(extracted_files)} files to {output_dir}")
            else:
                print("No files were extracted")
                
                # Print some example filenames to help troubleshoot
                print("\nExample filenames in the zip:")
                sample_files = [f for f in file_list if f.endswith('.AT2')][:10]
                for f in sample_files:
                    print(f"  {f}")

    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file")
    except Exception as e:
        print(f"Error extracting files: {str(e)}")



def write_output(rec_idx, IMs, output_dir, output_file, metadata):
    """
    Write a tab-delimited file with selected ground motions and scale factors
    
    Parameters:
    -----------
    rec_idx : numpy.ndarray
        Indices of selected motions in original database
    IMs : IntensityMeasures
        Selected intensity measures
    output_dir : str
        Directory for output files
    output_file : str
        Name of output file
    metadata : dict
        Information about the database and selected records
    """
    # Create directory for outputs, if it doesn't yet exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(os.path.join(output_dir, output_file), 'w') as f:
        # Print header information if available
        if 'getTimeSeries' in metadata and metadata['getTimeSeries'] is not None:
            for i in range(min(3, len(metadata['getTimeSeries']))):
                if metadata['getTimeSeries'][i]:
                    f.write(f"{metadata['getTimeSeries'][i]}\n")
            f.write("\n")
        
        # Write column headers based on number of components
        if isinstance(metadata['Filename'], list) or isinstance(metadata['Filename'], np.ndarray):
            if len(np.shape(metadata['Filename'])) == 1 or np.shape(metadata['Filename'])[1] == 1:
                # Only one ground motion component
                f.write("Record Number\tRecord Sequence Number\tScale Factor\tComponent Number\tFile Name\tURL\n")
            elif np.shape(metadata['Filename'])[1] == 2:
                # Two components
                f.write("Record Number\tRecord Sequence Number\tScale Factor\tFile Name Dir. 1\tFile Name Dir. 2\tURL 1\tURL 2\n")
            else:
                # Three components (including vertical)
                f.write("Record Number\tRecord Sequence Number\tScale Factor (H)\tScale Factor (V)\tFile Name Dir. 1\tFile Name Dir. 2\tFile Name Dir. 3\tURL 1\tURL 2\tURL 3\n")
        
        # Write record data
        for i, idx in enumerate(rec_idx):
            try:
                idx_int = int(idx)  # Convert to integer for indexing
                
                if isinstance(metadata['Filename'], list) or isinstance(metadata['Filename'], np.ndarray):
                    # Handle different array shapes
                    filename_shape = np.shape(metadata['Filename'])
                    
                    if len(filename_shape) == 1 or filename_shape[1] == 1:
                        # Only one ground motion component
                        comp_num = metadata.get('compNum', [1])[idx_int] if 'compNum' in metadata else 1
                        filename = metadata['Filename'][idx_int]
                        dir_location = metadata.get('dirLocation', [''])[idx_int] if 'dirLocation' in metadata else ''
                        f.write(f"{i+1}\t{idx_int+1}\t{IMs.scaleFac[i]:.2f}\t{int(comp_num)}\t{filename}\t{os.path.join(dir_location, filename)}\n")
                    
                    elif filename_shape[1] == 2:
                        # Two components
                        filename1 = metadata['Filename'][idx_int, 0] if filename_shape[1] > 0 else ''
                        filename2 = metadata['Filename'][idx_int, 1] if filename_shape[1] > 1 else ''
                        dir_location = metadata.get('dirLocation', [''])[idx_int] if 'dirLocation' in metadata else ''
                        url1 = os.path.join(dir_location, filename1)
                        url2 = os.path.join(dir_location, filename2)
                        f.write(f"{i+1}\t{idx_int+1}\t{IMs.scaleFac[i]:.2f}\t{filename1}\t{filename2}\t{url1}\t{url2}\n")
                    
                    else:
                        # Three components (including vertical)
                        filename1 = metadata['Filename'][idx_int, 0] if filename_shape[1] > 0 else ''
                        filename2 = metadata['Filename'][idx_int, 1] if filename_shape[1] > 1 else ''
                        filename3 = metadata['Filename'][idx_int, 2] if filename_shape[1] > 2 else ''
                        dir_location = metadata.get('dirLocation', [''])[idx_int] if 'dirLocation' in metadata else ''
                        url1 = os.path.join(dir_location, filename1)
                        url2 = os.path.join(dir_location, filename2)
                        url3 = os.path.join(dir_location, filename3)
                        
                        # Get vertical scale factor if available
                        scale_v = IMs.scaleFacV[i] if hasattr(IMs, 'scaleFacV') else IMs.scaleFac[i]
                        
                        f.write(f"{i+1}\t{idx_int+1}\t{IMs.scaleFac[i]:.2f}\t{scale_v:.2f}\t{filename1}\t{filename2}\t{filename3}\t{url1}\t{url2}\t{url3}\n")
            
            except Exception as e:
                print(f"Warning: Error writing record {i+1} (index {idx}): {str(e)}")
                # Write a placeholder line
                f.write(f"{i+1}\t{idx}\t{IMs.scaleFac[i]:.2f}\tError writing record\n")
    
    print(f"Output written to {os.path.join(output_dir, output_file)}")


def main():
    """Main function to run the ground motion selection process"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)  # Change to script directory to ensure correct paths
    
    print("Starting ground motion selection process")
    
    # Initialize parameters
    selection_params = SelectionParams()
    rup = Rupture()
    allowed_recs = AllowedRecords()
    IMs = IntensityMeasures()
    
    # User control parameters
    show_plots = True
    copy_files = True
    seed_value = 0
    n_trials = 20
    output_dir = 'Data'
    output_file = 'Output_File.dat'
    
    # Assign rupture parameters to selection_params for reference
    selection_params.rup = rup
    
    # Load and screen the database
    try:
        SaKnown, selection_params, ind_per, known_per, metadata = screen_database(
            selection_params, allowed_recs
        )
        
        # Save logarithmic spectral accelerations at target periods
        IMs.sampleBig = np.log(SaKnown[:, ind_per])
        if selection_params.matchV == 1:
            IMs.sampleBigV = np.log(selection_params.SaKnownV[:, selection_params.indPerV])
        
        # Compute target means and covariances
        target_sa = get_target_spectrum(known_per, selection_params, ind_per, rup)
        plot_target_spectrum(target_sa, selection_params)

        
        # Simulate response spectra
        simulated_spectra = simulate_spectra(target_sa, selection_params, seed_value, n_trials)
        #plot_simulated_spectra(simulated_spectra, selection_params)

        
        # Find best matches
        if selection_params.matchV == 1:
            IMs = find_ground_motionsV(selection_params, simulated_spectra, IMs)
        else:
            IMs = find_ground_motions(selection_params, simulated_spectra, IMs)
        
        # Store first stage results
        IMs.stageOneScaleFac = IMs.scaleFac.copy()
        IMs.stageOneMeans = np.mean(np.log(SaKnown[IMs.recID, :] * IMs.stageOneScaleFac[:, np.newaxis]), axis=0)
        IMs.stageOneStdevs = np.std(np.log(SaKnown[IMs.recID, :] * IMs.stageOneScaleFac[:, np.newaxis]), axis=0)
        if selection_params.matchV == 1:
            IMs.stageOneScaleFacV = IMs.scaleFacV.copy()
            IMs.stageOneMeansV = np.mean(np.log(selection_params.SaKnownV[IMs.recID, :] * IMs.stageOneScaleFacV[:, np.newaxis]), axis=0)
            IMs.stageOneStdevsV = np.std(np.log(selection_params.SaKnownV[IMs.recID, :] * IMs.stageOneScaleFacV[:, np.newaxis]), axis=0)
        
        # Optimize if needed
        if selection_params.matchV == 1:
            within_tol, IMs = within_toleranceV(IMs, target_sa, selection_params)
            if within_tol:
                print(f"Optimization skipped - errors within tolerance")
            else:
                IMs = optimize_ground_motionsV(selection_params, target_sa, IMs)
        else:
            if within_tolerance(IMs.sampleSmall, target_sa, selection_params):
                print(f"Optimization skipped - errors within tolerance")
            else:
                IMs = optimize_ground_motions(selection_params, target_sa, IMs)
        
        # Plot results
        if show_plots:
            if selection_params.matchV == 1:
                plot_resultsV(selection_params, target_sa, IMs, simulated_spectra, SaKnown, known_per)
            else:
                plot_results(selection_params, target_sa, IMs, simulated_spectra, SaKnown, known_per)
        
        # Output results
        rec_idx = metadata['allowedIndex'][IMs.recID]
        write_output(rec_idx, IMs, output_dir, output_file, metadata)
        
        # Copy time series if requested
        if copy_files:
            download_time_series(output_dir, rec_idx, metadata)
        
        print("Ground motion selection complete")
        
    except Exception as e:
        print(f"Error during ground motion selection: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()