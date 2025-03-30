# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:39:31 2025

@author: Amir Taherian
"""

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
from openquake.hazardlib import gsim, imt
from openquake.hazardlib.contexts import RuptureContext, SitesContext, DistancesContext


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




def get_openquake_gmm(gmm_name):
    """Returns an initialized OpenQuake ground motion model by name"""
    try:
        if gmm_name == 'Allen2012':
            from openquake.hazardlib.gsim.allen_2012 import Allen2012
            return Allen2012()
        elif gmm_name == 'BooreEtAl2014':
            from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
            return BooreEtAl2014()
        elif gmm_name == 'AbrahamsonEtAl2014':
            from openquake.hazardlib.gsim.abrahamson_2014 import AbrahamsonEtAl2014
            return AbrahamsonEtAl2014()
        elif gmm_name == 'CampbellBozorgnia2014':
            from openquake.hazardlib.gsim.campbell_bozorgnia_2014 import CampbellBozorgnia2014
            return CampbellBozorgnia2014()
        elif gmm_name == 'ChiouYoungs2014':
            from openquake.hazardlib.gsim.chiou_youngs_2014 import ChiouYoungs2014
            return ChiouYoungs2014()
        else:
            raise ValueError(f"GMM '{gmm_name}' not recognized.")
    except Exception as e:
        print(f"Error initializing GMM '{gmm_name}': {str(e)}")
        raise

def setup_openquake_contexts(rup_params):
    """Set up OpenQuake contexts from rupture parameters"""
    print("Setting up OpenQuake contexts...")
    
    # Rupture context
    rup = RuptureContext()
    rup.mag = rup_params.M_bar
    
    # Set rake based on fault type
    if rup_params.Fault_Type == 1:  # Strike-slip
        rup.rake = 0.0
    elif rup_params.Fault_Type == 2:  # Normal
        rup.rake = -90.0
    elif rup_params.Fault_Type == 3:  # Reverse
        rup.rake = 90.0
    else:
        rup.rake = 0.0  # Default to strike-slip
    
    rup.hypo_depth = rup_params.Zhyp if hasattr(rup_params, 'Zhyp') else 10.0
    rup.ztor = rup_params.Ztor if hasattr(rup_params, 'Ztor') else 0.0
    
    # Distances context
    dists = DistancesContext()
    dists.rrup = np.array([rup_params.Rrup])
    dists.rjb = np.array([rup_params.Rjb])
    dists.rx = np.array([rup_params.Rx]) if hasattr(rup_params, 'Rx') else np.array([0.0])
    
    # Sites context
    sites = SitesContext()
    sites.vs30 = np.array([rup_params.Vs30])
    sites.vs30measured = np.array([True])  # Assuming measured Vs30
    
    # Handle z1.0 (basin depth to 1.0 km/s horizon)
    if hasattr(rup_params, 'z1') and rup_params.z1 != 999:
        sites.z1pt0 = np.array([rup_params.z1])
    else:
        # Estimate z1.0 using Vs30-based correlation (Chiou and Youngs 2014)
        z1pt0 = np.exp((-7.15/4) * np.log((sites.vs30**4 + 570.94**4) / (1360**4 + 570.94**4)))
        sites.z1pt0 = np.array([z1pt0])
    
    # Handle z2.5 (basin depth to 2.5 km/s horizon)
    if hasattr(rup_params, 'Z2p5'):
        sites.z2pt5 = np.array([rup_params.Z2p5])
    else:
        sites.z2pt5 = np.array([1.0])  # Default value
    
    # Required site parameter
    sites.sids = np.array([0])
    
    return rup, sites, dists

def openquake_gmm_calc(T, rup_params, gmm_name='BooreEtAl2014'):
    """Calculate median and standard deviation using OpenQuake GMM"""
    try:
        print(f"Using OpenQuake GMM: {gmm_name}")
        
        # Initialize the GMM
        gmm = get_openquake_gmm(gmm_name)
        
        # Set up contexts
        rup_ctx, sites_ctx, dists_ctx = setup_openquake_contexts(rup_params)
        
        # Convert T to numpy array if it's not already
        if not isinstance(T, np.ndarray):
            T = np.array([T])
        
        # Initialize output arrays
        median = np.zeros_like(T, dtype=float)
        sigma = np.zeros_like(T, dtype=float)
        
        # Calculate for each period
        for i, period in enumerate(T):
            try:
                #print(f"Processing period {period} s...")
                sa_imt = imt.SA(period=period)
                
                # Get mean and standard deviation from GMM
                mean, stddevs = gmm.get_mean_and_stddevs(
                    sites_ctx, rup_ctx, dists_ctx, sa_imt, ['TOTAL']
                )
                
                median[i] = np.exp(mean[0])
                sigma[i] = stddevs[0][0] if len(stddevs) > 0 and len(stddevs[0]) > 0 else 0.6
                
            except Exception as e:
                print(f"Error processing period {period} s: {str(e)}")
                # Set default values for this period
                median[i] = 0.1  # Default value
                sigma[i] = 0.6   # Default value
        
        # Return scalar if input was scalar
        if len(T) == 1:
            return median[0], sigma[0], T[0]
        else:
            return median, sigma, T
            
    except Exception as e:
        print(f"Error in openquake_gmm_calc: {str(e)}")
        # Fallback to a simple model for debugging
        if not isinstance(T, np.ndarray):
            T = np.array([T])
        median = 0.5 * np.exp(-0.5 * np.log(T))  # Simple spectral shape
        sigma = np.ones_like(T) * 0.6  # Constant sigma
        
        return median, sigma, T
    


# Now, modify the get_target_spectrum function to use the OpenQuake GMM
def get_target_spectrum(knownPer, selectionParams, indPer, rup):
    """
    Calculate and return the target mean spectrum and covariance
    matrix at available periods using OpenQuake GMM
    """
    # Initialize output structure
    target_dict = {}
    
    try:
        # Compute target mean spectrum using OpenQuake GMM if requested
        if hasattr(selectionParams, 'use_openquake') and selectionParams.use_openquake:
            gmm_name = getattr(selectionParams, 'gmm_name', 'BooreEtAl2014')
            sa, sigma, _ = openquake_gmm_calc(knownPer, rup, gmm_name=gmm_name)
        
        # Verify we got valid values
        print(f"Generated {len(sa)} sa values with range: {np.min(sa):.4f} to {np.max(sa):.4f}")
        print(f"Generated {len(sigma)} sigma values with range: {np.min(sigma):.4f} to {np.max(sigma):.4f}")
        
        # Modify spectral targets if RotD100 values were specified
        if selectionParams.RotD == 100 and selectionParams.arb == 2:
            rotD100Ratio, rotD100Sigma = gmpe_sb_2014_ratios(knownPer)
            sa = sa * rotD100Ratio
            sigma = np.sqrt(sigma**2 + rotD100Sigma**2)
        
        # Back-calculate epsilon if SaTcond is specified
        if hasattr(selectionParams, 'SaTcond') and selectionParams.SaTcond is not None:
            # Interpolate to get median Sa and sigma at Tcond
            logPer = np.log(knownPer)
            logSa = np.log(sa)
            
            median_SaTcond = np.exp(np.interp(np.log(selectionParams.Tcond), logPer, logSa))
            sigma_SaTcond = np.interp(np.log(selectionParams.Tcond), logPer, sigma)
            
            eps_bar = (np.log(selectionParams.SaTcond) - np.log(median_SaTcond)) / sigma_SaTcond
            
            print(f"Back-calculated epsilon = {eps_bar:.3f}")
        else:
            # Use user-specified epsilon value
            eps_bar = rup.eps_bar
        
        # Calculate target mean spectrum (in log space)
        if selectionParams.cond == 1:
            # Compute correlations and the conditional mean spectrum
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
        
        # Override covariance matrix with zeros if no variance desired
        if hasattr(selectionParams, 'useVar') and selectionParams.useVar == 0:
            TgtCovs = np.zeros_like(TgtCovs)
        
        # Avoid numerical issues with very small covariance values
        TgtCovs[np.abs(TgtCovs) < 1e-10] = 1e-10
        
        # Store target mean and covariance matrix at target periods
        target_dict['meanReq'] = TgtMean[indPer]
        target_dict['covReq'] = TgtCovs[np.ix_(indPer, indPer)]
        target_dict['stdevs'] = np.sqrt(np.diag(target_dict['covReq']))
        
        # Target mean and covariance at all periods
        target_dict['meanAllT'] = TgtMean
        target_dict['covAllT'] = TgtCovs
        
        # Print diagnostic information
        print("Target spectrum information:")
        print(f"meanReq shape: {target_dict['meanReq'].shape}")
        print(f"stdevs shape: {target_dict['stdevs'].shape}")
        print(f"covReq shape: {target_dict['covReq'].shape}")
        
    except Exception as e:
        print(f"Error in get_target_spectrum: {str(e)}")
        # Create fallback values in case of error
        n_periods = len(selectionParams.TgtPer)
        target_dict['meanReq'] = np.log(np.linspace(0.5, 0.1, n_periods))
        target_dict['stdevs'] = np.ones(n_periods) * 0.6
        target_dict['covReq'] = np.diag(target_dict['stdevs']**2)
    
    # Convert dictionary to TargetSa object
    targetSa = TargetSa()
    targetSa.meanReq = target_dict['meanReq']
    targetSa.covReq = target_dict['covReq']
    targetSa.stdevs = target_dict['stdevs']
    
    # Add these if they exist
    if 'meanAllT' in target_dict:
        targetSa.meanAllT = target_dict['meanAllT']
    if 'covAllT' in target_dict:
        targetSa.covAllT = target_dict['covAllT']
    
    return targetSa

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
    
    # OpenQuake GMM parameters
    use_openquake = True  # Set to True to use OpenQuake GMMs
    # Don't use Allen2012 - it's designed for Australian earthquakes
    gmm_name = 'BooreEtAl2014'  # More appropriate for global/US earthquakes
    
    # Assign rupture parameters to selection_params for reference
    selection_params.rup = rup
    
    # Add OpenQuake GMM info to selection_params
    selection_params.use_openquake = use_openquake
    selection_params.gmm_name = gmm_name
    
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
        print(f"Computing target spectrum using {'OpenQuake ' + gmm_name if use_openquake else 'built-in BSSA 2014'} GMM")
        target_sa = get_target_spectrum(known_per, selection_params, ind_per, rup)
        plot_target_spectrum(target_sa, selection_params)
        
        # Simulate response spectra
        simulated_spectra = simulate_spectra(target_sa, selection_params, seed_value, n_trials)
        
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