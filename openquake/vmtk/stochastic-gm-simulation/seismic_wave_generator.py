# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:39:31 2025

@author: Amir Taherian
"""
import os
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List, Optional
from scipy.fft import fft, ifft
from scipy import signal
from scipy import linalg
from scipy.signal import butter, sosfilt, savgol_filter
from models import amfBJ


@dataclass
class GeometricSpreadingParameters:
    r_ref: float = 1.0
    segments: List[Tuple[float, float]] = field(default_factory=lambda: [(1.0, -1.0)])
    
    def get_spreading(self, r: float) -> float:
        if r <= self.r_ref:
            return (r/self.r_ref)**self.segments[0][1]
        
        g = (self.r_ref/self.r_ref)**self.segments[0][1]
        prev_r = self.r_ref
        
        for i, (dist, slope) in enumerate(self.segments):
            if r <= dist:
                g *= (r/prev_r)**slope
                return g
            elif i < len(self.segments) - 1:
                g *= (dist/prev_r)**slope
                prev_r = dist
            else:
                g *= (r/prev_r)**slope
                return g
        return g

@dataclass
class QualityFactorParameters:
    Q0: float = 100.0
    eta: float = 0.5
    Qmin: float = 100.0
    
    def get_Q(self, f: np.ndarray) -> np.ndarray:
        return np.maximum(self.Q0 * f**self.eta, self.Qmin)
    
@dataclass
class PathDurationParameters:
    """Parameters defining the path duration model."""
    duration_points: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (10.0, 0.5)])
    slope_beyond_last: float = 0.05  # Duration increase per km beyond last point
    
    def get_duration(self, r: float) -> float:
        """Compute path duration for distance r."""
        if r <= self.duration_points[0][0]:
            return self.duration_points[0][1]
        
        for i in range(1, len(self.duration_points)):
            r1, d1 = self.duration_points[i-1]
            r2, d2 = self.duration_points[i]
            
            if r <= r2:
                # Linear interpolation between defined duration points
                return d1 + (d2 - d1) * (r - r1) / (r2 - r1)
        
        # Beyond the last point, use the specified slope
        last_r, last_d = self.duration_points[-1]
        return last_d + self.slope_beyond_last * (r - last_r)

@dataclass
class SiteAttenuationParameters:
    kappa: float = 0.03
    
    def get_diminution(self, f: np.ndarray) -> np.ndarray:
        return np.exp(-np.pi * f * self.kappa)
    
@dataclass
class PulseParameters:
    """Parameters for the Mavroeidis and Papageorgiou pulse model."""
    enabled: bool = False
    gamma: float = 0.5
    nu: float = 0.0
    t0: Optional[float] = None
    peak_factor: float = 1.0

@dataclass
class StochasticModelParameters:
    """Complete set of parameters for stochastic ground motion simulation."""
    # Basic simulation parameters
    dt: float = 0.005  # Time step (s)
    ns: int = 5        # Number of simulations
    
    # Ground motion model components
    geometric_spreading: GeometricSpreadingParameters = field(default_factory=GeometricSpreadingParameters)
    quality_factor: QualityFactorParameters = field(default_factory=QualityFactorParameters)
    path_duration: PathDurationParameters = field(default_factory=PathDurationParameters)
    site_attenuation: SiteAttenuationParameters = field(default_factory=SiteAttenuationParameters)
    
    # Source parameters
    stress_drop: float = 200.0  # Stress parameter (bars)
    
    # Crustal parameters
    roll: float = 2.8    # Density (g/cm³)
    beta: float = 3.5  # Shear-wave velocity (km/s)
    
    # Site parameters
    vs30: float = 760.0  # Average shear-wave velocity in the top 30m (m/s)
    
    # Additional control parameters
    taper_percentage: float = 0.05  # Taper percentage for time window
    high_cut_freq: float = 50.0     # High-cut filter frequency (Hz)
    high_cut_order: int = 8         # High-cut filter order

@dataclass
class EarthquakeParameters:
    M: float
    rake: float
    strike: float
    dip: float
    h_ref: float
    stress_ref: float
    sigma: float

@dataclass
class SimulationParameters:
    # Original parameters
    NS: int
    dt: float
    roll: float
    beta: float
    Vs30: float
    Tr: np.ndarray
    kappa: float
    tpad: float
    #iflagscalefactor: int
    pulsing_percent: float
    rupture_velocity: float
    pulse_params: PulseParameters = field(default_factory=PulseParameters)

    
    # New parameter to hold ground motion model settings
    gm_model: StochasticModelParameters = None  # Default to None, user must provide

@dataclass
class SiteParameters:
    site_lat: float
    site_lon: float

@dataclass
class FaultParameters:
    subfault_size: float
    rupture_lat: float
    rupture_lon: float





SCALING_ACCELERATION = 1  # iflagscalefactor value for acceleration scaling
SCALING_BOORE = 2  # iflagscalefactor value for David Boore's scaling
MOMENT_CONSTANT = 16.05 # Constant in seismic moment calculation
FREQUENCY_CONSTANT = 4.906e6 # Constant in static corner frequency calc

# 1. Fault Grid Initialization
def initialize_fault_grid(earthquake_params: EarthquakeParameters, fault_params: FaultParameters) -> Tuple[float, float, int, int, float]:
    """Initializes parameters related to the fault grid."""
    fault_width, fault_length = width_length(earthquake_params.M, earthquake_params.rake, earthquake_params.sigma, earthquake_params.stress_ref)
    num_subfaults_length = max(1, int(fault_length / fault_params.subfault_size))
    num_subfaults_width = max(1, int(fault_width / fault_params.subfault_size))
    subRadius = np.sqrt((fault_params.subfault_size**2) / np.pi)  # Subfault radius
    return fault_width, fault_length, num_subfaults_length, num_subfaults_width, subRadius

# 2. Generate Hypocenters
def generate_hypocenters(Nhyp: int, fault_params: FaultParameters, earthquake_params: EarthquakeParameters,
                         fault_width: float, fault_length: float, num_subfaults_length: int,
                         num_subfaults_width: int, site_params: SiteParameters) -> np.ndarray:
    """Generates multiple hypocenter locations."""
    return calculate_random_hypocenters(
        Nhyp, fault_params.rupture_lat, fault_params.rupture_lon, earthquake_params.h_ref, earthquake_params.strike, earthquake_params.dip, fault_length,
        fault_width, num_subfaults_length, num_subfaults_width, site_params.site_lat, site_params.site_lon
    )

# 3. Compute Seismic Moment
def compute_seismic_moment(earthquake_params: EarthquakeParameters,simulation_params: SimulationParameters, num_subfaults: int) -> Tuple[float, float, float, float]:
    """Computes seismic moment-related parameters."""
    M0 = 10 ** (1.5 * earthquake_params.M + MOMENT_CONSTANT)  # Total seismic moment (dyne·cm)
    averageMoment = M0 / num_subfaults
    fc_static = FREQUENCY_CONSTANT * simulation_params.beta * ((earthquake_params.sigma / M0) ** (1 / 3))
    fc_st = FREQUENCY_CONSTANT * simulation_params.beta * ((earthquake_params.sigma / averageMoment) ** (1 / 3))
    return M0, averageMoment, fc_static, fc_st

def initialize_time_frequency(simulation_params: SimulationParameters, earthquake_params: EarthquakeParameters, subTrise, subTarrive, subTend) -> Tuple[np.ndarray, int, float, float, float]:
    """
    Initializes time and frequency-related parameters to match GMSS2SS.m.
    """
    #Constants and Moment Calculation:
    C = (0.55 * 2.0 * 0.707) * 1e-20 / (4 * np.pi * simulation_params.roll * simulation_params.beta**3)

    M0=10**(1.5*earthquake_params.M+16.05)                               # total seismic moment
    
    # Find maximum Trise, minimum Tarrive and maximum Tend
    Trise_max = np.max(subTrise)
    Tarrive_min = np.min(subTarrive)
    Tend_max = np.max(subTend)

    tpadl, tpadt = simulation_params.tpad, simulation_params.tpad
    # Calculate NtotalWave
    NtotalWave = (Tend_max - Tarrive_min + tpadl + tpadt + Trise_max) / simulation_params.dt

    # Calculate NT (nearest power of 2)
    nseed = round(np.log2(NtotalWave))
    if 2**nseed >= NtotalWave:
        NT = 2**nseed
    else:
        NT = 2**(nseed + 1)

    # Calculate T, df, fmax
    T = NT * simulation_params.dt
    df = 1 / T
    fmax = 1 / (2 * simulation_params.dt)

    # Create time vector
    t = np.arange(simulation_params.dt, T + simulation_params.dt, simulation_params.dt)

    return t, NT, df, fmax,C,M0


# 5. Initialize Rupture Variables
class RuptureParameters:
    """A simple class to encapsulate rupture parameters."""
    def __init__(self, R_epicentral: float, h_ref: float, dip: float, strike: float, Az: float):
        """Initializes rupture parameters."""
        self.R_epicentral = R_epicentral
        self.h_ref = h_ref
        self.dip = dip
        self.strike = strike
        self.Az = Az

def process_single_subfault(i: int, j: int, i0: int, j0: int, dl: float, dw: float,
                          simulation_params: SimulationParameters, beta: float, subRadius: float,
                          rpathdur: list, pathdur: list, durslope: float, no_effective_subfaults: int,
                          rupture_params: RuptureParameters, active_subfaults: np.ndarray,
                          num_subfaults_length: int, num_subfaults_width: int) -> Tuple[
                              int, float, float, float, float, float, float]:
    
    # Convert 0-based indices to 1-based indices for use in utils functions
    i_onebased = i + 1
    j_onebased = j + 1
    i0_onebased = i0 + 1
    j0_onebased = j0 + 1

    # Calculate n_pulsing_subs but don't use it to determine activity
    n_pulsing_subs = number_of_pulsing_subs(
        i_onebased, j_onebased, i0_onebased, j0_onebased, num_subfaults_length, num_subfaults_width, no_effective_subfaults
    )
    
    # ALL subfaults are active in EXSIM
    active_subfaults[i, j] = 1
    
    # Store n_pulsing_subs for later use in corner frequency calculation
    # You might need to return it or store it in a global array
    
    # Calculate subfault parameters normally
    R_subfault = compute_subfault_distance(
        rupture_params.R_epicentral, rupture_params.h_ref, rupture_params.dip, rupture_params.strike, rupture_params.Az, dl, dw, i_onebased, j_onebased
    )
    
    delay_val = np.sqrt((dl * (i_onebased - i0_onebased)) ** 2 + (dw * (j_onebased - j0_onebased)) ** 2) / simulation_params.rupture_velocity if (i_onebased != i0_onebased or j_onebased != j0_onebased) else 0.0
    t_arrive_val = delay_val + R_subfault / simulation_params.beta
    rise_time_val = subRadius / simulation_params.rupture_velocity
    dur_path_val = compute_t_path(R_subfault, rpathdur, pathdur, durslope)
    dur_sub_val = dur_path_val + rise_time_val
    
    return active_subfaults[i, j], R_subfault, delay_val, t_arrive_val, rise_time_val, dur_path_val, dur_sub_val



def process_subfaults(num_subfaults_length: int, num_subfaults_width: int, i0: int, j0: int, dl: float, dw: float,
                          simulation_params: SimulationParameters, beta: float, subRadius: float, rpathdur: list, pathdur: list, durslope: float,
                          no_effective_subfaults: int, rupture_params: RuptureParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Processes all subfaults in the grid."""
    active_subfaults = np.zeros((num_subfaults_length, num_subfaults_width), dtype=int)
    R_subfaults = np.zeros((num_subfaults_length, num_subfaults_width))
    delay = np.zeros((num_subfaults_length, num_subfaults_width))
    t_arrive = np.zeros((num_subfaults_length, num_subfaults_width))
    rise_time = np.zeros((num_subfaults_length, num_subfaults_width))
    dur_path = np.zeros((num_subfaults_length, num_subfaults_width))
    dur_sub = np.zeros((num_subfaults_length, num_subfaults_width))

    t_arrive_min = float("inf")
    t_end_max = 0

    active_subfaults[i0, j0] = 1

    for i in range(num_subfaults_length):
        for j in range(num_subfaults_width):
            active, R_subfault, delay_val, t_arrive_val, rise_time_val, dur_path_val, dur_sub_val = process_single_subfault(
                i, j, i0, j0, dl, dw, simulation_params, beta, subRadius,
                rpathdur, pathdur, durslope, no_effective_subfaults, # PASSED ARGUMENTS
                rupture_params, active_subfaults, num_subfaults_length, num_subfaults_width #PASSED ARGUMENTS
            )
            

            active_subfaults[i, j] = active
            R_subfaults[i, j] = R_subfault
            delay[i, j] = delay_val
            t_arrive[i, j] = t_arrive_val #if (np.isnan(t_arrive_val) == False and active == True) else 0
            rise_time[i, j] = rise_time_val
            dur_path[i, j] = dur_path_val
            dur_sub[i, j] = dur_sub_val

            if active == True and np.isnan(t_arrive_val) == False:
                t_arrive_min = min(t_arrive_min, t_arrive[i, j])
                t_end_max = max(t_arrive[i, j] + dur_sub[i, j], t_end_max)
            else:
                 t_arrive[i, j] = 0
                 rise_time[i, j] = 0 #To ensure to prevent error
                 dur_path[i,j] = 0 #To ensure to prevent error
                 dur_sub[i,j] = 0 #To ensure to prevent error

    #  Validate active subfaults
    assert np.any(active_subfaults), "No active subfaults found. Check pulsing logic or input parameters."

    if t_arrive_min == float("inf"):
        raise ValueError("t_arrive_min was not updated. Ensure at least one subfault is active and arrival times are calculated.")

    return (
        active_subfaults, R_subfaults, delay, t_arrive, rise_time,
        dur_path, dur_sub, t_arrive_min, t_end_max
    )

# Distance calculation functions

def width_length(mag, rake,stress,stress_ref):
    assert -180 <= rake <= 180, 'Rake must be between -180 and 180.'

    if rake is None:
        faultwidth = 10 ** (-1.01 + 0.32 * mag)*(stress_ref/stress)**(1/3)
        faultlength = 10 ** (-2.44 + 0.59 * mag)*(stress_ref/stress)**(1/3)
    elif (-45 <= rake <= 45) or rake >= 135 or rake <= -135:
        faultwidth = 10 ** (-0.76 + 0.27 * mag)*(stress_ref/stress)**(1/3)
        faultlength = 10 ** (-2.57 + 0.62 * mag)*(stress_ref/stress)**(1/3)
    elif rake > 0:
        faultwidth = 10 ** (-1.61 + 0.41 * mag)*(stress_ref/stress)**(1/3)
        faultlength = 10 ** (-2.42 + 0.58 * mag)*(stress_ref/stress)**(1/3)
    else:
        faultwidth = 10 ** (-1.14 + 0.35 * mag)*(stress_ref/stress)**(1/3)
        faultlength = 10 ** (-1.88 + 0.50 * mag)*(stress_ref/stress)**(1/3)

    return faultwidth, faultlength



def haversine_distance(lat1, lon1, lat2, lon2):
    radius_earth = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius_earth * np.arcsin(np.sqrt(a))

def compute_subfault_distance(R_hypo, h_ref, Fdip, Fstrike, Az, dl, dw, i, j):
    """
    Python equivalent of FUNsubR: computes the distance between a site and a subfault.

    Parameters:
    - R_hypo: Epicentral distance (km)
    - h_ref: Depth of fault reference point (km)
    - Fdip: Fault dip angle (degrees)
    - Fstrike: Fault strike angle (degrees)
    - Az: Azimuth to the site (degrees)
    - dl: Along-strike subfault length (km)
    - dw: Down-dip subfault width (km)
    - i, j: Subfault indices (along strike and dip)

    Returns:
    - subR: Distance to subfault (km)
    """

    # Convert angles to radians
    Fstrike_radians = np.radians(Az - Fstrike)
    Fdip_radians = np.radians(90 - Fdip)

    # Compute offsets
    t1 = R_hypo * np.cos(Fstrike_radians) - ((2 * i - 1) * dl / 2)
    t2 = R_hypo * np.sin(Fstrike_radians) - ((2 * j - 1) * dw / 2) * np.sin(Fdip_radians)
    t3 = -h_ref - ((2 * j - 1) * dw / 2) * np.cos(Fdip_radians)

    # Euclidean distance
    subR = np.sqrt(t1**2 + t2**2 + t3**2)
    return subR
    
def compute_site_location(SLfactor, SL1, SL2, FaultLat, FaultLon):
    """
    Computes the site location's epicentral distance and azimuth relative to the fault origin.

    Parameters:
    - SLfactor: Determines input type (1 for lat/lon; 2 for distance/azimuth).
    - SL1, SL2: Inputs (latitude/longitude or distance/azimuth).
    - FaultLat, FaultLon: Latitude and longitude of the fault origin.

    Returns:
    - SiteLat: Latitude of the site (if applicable).
    - SiteLon: Longitude of the site (if applicable).
    - R: Epicentral distance (km).
    - Az: Azimuth (degrees).
    """

    if SLfactor == 1:
        # Input is in lat/lon
        site_lat = SL1
        site_lon = SL2
        
        # Calculate epicentral distance (R)
        # Haversine distance for epicentral distance
        R = haversine_distance(site_lat, site_lon, FaultLat, FaultLon)
            
        # Calculate azimuth (Az)
        delta_lon = np.radians(site_lon - FaultLon)
        lat1, lat2 = np.radians(FaultLat), np.radians(site_lat)

        x = np.sin(delta_lon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
        Az = (np.degrees(np.arctan2(x, y)) + 360) % 360  # Normalize to [0, 360)

    elif SLfactor == 2:
        # Input is R (distance) and Az (azimuth)
        R = SL1
        Az = SL2
        
        # Convert back to lat/lon if needed
        earth_radius = 6371  # Earth's radius in km
        angular_distance = R / earth_radius
        azimuth_rad = np.radians(Az)
        
        FaultLat_rad = np.radians(FaultLat)
        FaultLon_rad = np.radians(FaultLon)
        
        site_lat = np.degrees(np.arcsin(np.sin(FaultLat_rad) * np.cos(angular_distance) +
                                       np.cos(FaultLat_rad) * np.sin(angular_distance) * np.cos(azimuth_rad)))
        site_lon = np.degrees(FaultLon_rad + np.arctan2(np.sin(azimuth_rad) * np.sin(angular_distance) * np.cos(FaultLat_rad),
                                                       np.cos(angular_distance) - np.sin(FaultLat_rad) * np.sin(np.radians(site_lat))))
    else:
        raise ValueError("SLfactor must be 1 (lat/lon input) or 2 (distance/azimuth input).")

    return site_lat, site_lon, R, Az

# First, add these two functions to your imports section
def calculate_FUNh(rx, rz, w1, w2, s1, s2):
    """Calculate distance to fault rupture (helper function)"""
    if rx <= s1:
        if rz <= w1:
            h = np.sqrt((rx - s1)**2 + (rz - w1)**2)
            icase = 1
        elif rz > w1 and rz < w2:
            h = np.abs(rx - s1)
            icase = 4
        else:  # rz >= w2
            h = np.sqrt((rx - s1)**2 + (rz - w2)**2)
            icase = 7
    elif rx > s1 and rx < s2:
        if rz <= w1:
            h = np.abs(rz - w1)
            icase = 2
        elif rz > w1 and rz < w2:
            h = 0  # Inside the rupture plane
            icase = 5
        else:  # rz >= w2
            h = np.abs(rz - w2)
            icase = 8
    else:  # rx >= s2
        if rz <= w1:
            h = np.sqrt((rx - s2)**2 + (rz - w1)**2)
            icase = 3
        elif rz > w1 and rz < w2:
            h = np.abs(rx - s2)
            icase = 6
        else:  # rz >= w2
            h = np.sqrt((rx - s2)**2 + (rz - w2)**2)
            icase = 9
    return h, icase

def calculate_fault_distances(site_lat, site_lon, ref_lat, ref_lon, f_strike, f_dip, h_ref, 
                             fault_length, fault_width):
    """Calculate Rrup, Rjb and other fault distances"""
        # Define fault boundaries for distance calculations
    s1 = -fault_length/2
    s2 = fault_length/2
    w1 = 0  # Top of fault 
    w2 = fault_width
    h_min = 0  # Minimum seismogenic depth

    d2r = np.pi / 180.0
    fstrike = f_strike * d2r
    fdip = f_dip * d2r
    
    # Compute unit vectors
    ix_n = np.cos(fstrike)
    ix_e = np.sin(fstrike)
    ix_d = 0
    iy_n = -np.sin(fstrike) * np.sin(fdip)
    iy_e = np.cos(fstrike) * np.sin(fdip)
    iy_d = -np.cos(fdip)
    iz_n = -np.sin(fstrike) * np.cos(fdip)
    iz_e = np.cos(fstrike) * np.cos(fdip)
    iz_d = np.sin(fdip)
    
    # Convert site lat/lon to northern and eastern distances
    dist_n = (site_lat - ref_lat) * d2r * 6371
    dist_e = (site_lon - ref_lon) * d2r * np.cos(0.5 * d2r * (site_lat + ref_lat)) * 6371
    dist_d = -h_ref
    
    # Convert coordinates to fault reference frame
    rx = dist_n * ix_n + dist_e * ix_e + dist_d * ix_d
    ry = dist_n * iy_n + dist_e * iy_e + dist_d * iy_d
    rz = dist_n * iz_n + dist_e * iz_e + dist_d * iz_d
    
    # Find Rrup
    hrup, _ = calculate_FUNh(rx, rz, w1, w2, s1, s2)
    Rrup = np.sqrt(hrup**2 + ry**2)
            
    # Find Rjb
    s1jb = s1
    s2jb = s2
    w1jb = w1 * np.cos(fdip)
    w2jb = w2 * np.cos(fdip)
    rxjb = rx
    rzjb = -np.sin(fstrike) * dist_n + np.cos(fstrike) * dist_e
    
    hjb, _ = calculate_FUNh(rxjb, rzjb, w1jb, w2jb, s1jb, s2jb)
    Rjb = hjb
        
    return Rrup, Rjb


def compute_point_source_distance(R, h_ref, Fdip, Fstrike, Az):
    """
    Computes the distance between a site and a point source (hypocenter).

    Parameters:
    - R: Epicentral distance (km).
    - h_ref: Depth of the point source (km).
    - Fdip: Fault dip angle (degrees).
    - Fstrike: Fault strike angle (degrees).
    - Az: Azimuth to the site (degrees).

    Returns:
    - pointR: Distance to the point source (km).
    """

    # Convert angles to radians
    Fstrike_radians = np.radians(Az - Fstrike)
    Fdip_radians = np.radians(90 - Fdip)

    # Hypocenter position (simplified for a point source)
    t1 = R * np.cos(Fstrike_radians)
    t2 = R * np.sin(Fstrike_radians) * np.sin(Fdip_radians)
    t3 = -h_ref - R * np.sin(Fstrike_radians) * np.cos(Fdip_radians)

    # Euclidean distance
    pointR = np.sqrt(t1**2 + t2**2 + t3**2)
    return pointR


def number_of_pulsing_subs(i, j, i0, j0, num_subfaults_length, num_subfaults_width, no_effective_subfaults):
    """
    Determines the number of pulsing subfaults around the given subfault (i, j).
    Adjusted to align with MATLAB's FUNNP logic.
    """
    # Compute Rmax and Rmin
    r_max = max(abs(i - i0) + 1, abs(j - j0) + 1)
    r_min = max(0, r_max - no_effective_subfaults)

    n = 0  # Counter for active subfaults

    # Iterate over the entire fault grid (adjusted for MATLAB indexing)
    for ii in range(1, num_subfaults_length + 1):
        for jj in range(1, num_subfaults_width + 1):
            # Adjust i0, j0 for MATLAB-style indexing
            r = max(abs(ii - (i0 + 1)) + 1, abs(jj - (j0 + 1)) + 1)
            if r_min < r < r_max:  # Adjust condition to match MATLAB
                n += 1

    return n

def locate_hypocenter_randomly(fault_length, fault_width, num_length, num_width):
    """
    Randomly selects a hypocenter on the fault plane based on subfault dimensions.

    Parameters:
    fault_length (float): Total fault length (km).
    fault_width (float): Total fault width (km).
    num_length (int): Number of subfaults along the length.
    num_width (int): Number of subfaults along the width.

    Returns:
    tuple: Hypocenter coordinates (x, y) in fault plane indices.
    """
    # Randomly select indices within the valid range
    i0 = np.random.randint(0, num_length)  # Python indexing: 0 <= i0 < num_length
    j0 = np.random.randint(0, num_width)  # Python indexing: 0 <= j0 < num_width
    
    return i0, j0


def calculate_random_hypocenters(
    Nhyp, rupture_lat, rupture_lon, h_ref, strike, dip, fault_length, fault_width,
    num_length, num_width, site_lat=None, site_lon=None
):
    """
    Calculates multiple random hypocenters and their distances and azimuths relative to the fault's top edge.

    Parameters:
    - Nhyp (int): Number of hypocenters to generate.
    - rupture_lat (float): Fault's top-center latitude.
    - rupture_lon (float): Fault's top-center longitude.
    - h_ref (float): Fault's top depth (km).
    - strike (float): Fault strike angle (degrees).
    - dip (float): Fault dip angle (degrees).
    - fault_length (float): Total fault length (km).
    - fault_width (float): Total fault width (km).
    - num_length (int): Number of subfaults along the length.
    - num_width (int): Number of subfaults along the width.
    - site_lat (float, optional): Site latitude (degrees).
    - site_lon (float, optional): Site longitude (degrees).

    Returns:
    - list: A list of dictionaries, each representing a hypocenter's properties:
        - 'latitude': Latitude of the hypocenter.
        - 'longitude': Longitude of the hypocenter.
        - 'depth': Depth of the hypocenter (km).
        - 'R': Distance to the site (if site location is provided).
        - 'Az': Azimuth to the site (if site location is provided).
        - 'i0': Hypocenter's subfault index along the length.
        - 'j0': Hypocenter's subfault index along the width.
    """
    hypocenters = []

    for _ in range(Nhyp):
        # Generate a single random hypocenter
        i0, j0 = locate_hypocenter_randomly(fault_length, fault_width, num_length, num_width)
        
        # Calculate subfault dimensions
        subfault_length = fault_length / num_length
        subfault_width = fault_width / num_width

        # Hypocenter's local coordinates on the fault plane
        x_offset = (i0 - 0.5) * subfault_length  # Middle of subfault in the x-direction
        z_offset = (j0 - 0.5) * subfault_width   # Middle of subfault in the z-direction (down-dip)

        # Convert to geographical coordinates
        dx = x_offset * np.cos(np.radians(strike))
        dy = x_offset * np.sin(np.radians(strike))
        dz = z_offset * np.sin(np.radians(dip))

        hypo_lat = rupture_lat + (dy / 111)  # Approximate conversion of km to degrees latitude
        hypo_lon = rupture_lon + (dx / (111 * np.cos(np.radians(rupture_lat))))  # Adjust for longitude convergence
        hypo_depth = h_ref + dz

        # Calculate site distance and azimuth (if site location is provided)
        if site_lat is not None and site_lon is not None:
            delta_lat = np.radians(site_lat - rupture_lat)
            delta_lon = np.radians(site_lon - rupture_lon)
            avg_lat = np.radians((site_lat + rupture_lat) / 2)

            R = np.sqrt((delta_lat * 6371) ** 2 + (delta_lon * 6371 * np.cos(avg_lat)) ** 2)
            Az = np.degrees(np.arctan2(delta_lon, delta_lat)) % 360
        else:
            R, Az = None, None

        # Store hypocenter properties
        hypocenters.append({
            "latitude": hypo_lat,
            "longitude": hypo_lon,
            "depth": hypo_depth,
            "R": R,
            "Az": Az,
            "i0": i0,
            "j0": j0,
        })

    return hypocenters

# Central Difference Method 
def calc_psa_cdm(at, dt, Tr, psi=0.05):
    """
    Implementation of the Central Difference Method (CDM) for PSA calculation.
    Converted from the provided MATLAB code.
    
    Parameters:
    at : numpy array
        Acceleration time series
    dt : float
        Time step
    Tr : numpy array
        Natural periods
    psi : float, optional
        Damping ratio (default is 0.05)
        
    Returns:
    RSD, RSV, RSA : numpy arrays
        Spectral displacement, velocity, and acceleration
    """
    # Calculate frequencies and parameters
    fr = 1.0 / Tr
    wn = fr * (2 * np.pi)
    A = -wn**2 + 2/(dt**2)
    B = psi * wn / dt - 1/(dt**2)
    C = 1/(dt**2) + psi * wn / dt
    
    # Initialize
    Nr = len(Tr)
    NT = len(at)
    u = np.zeros((Nr, NT+1))  # +1 for the extra step in CDM
    
    # Set initial displacement
    u[:, 0] = np.zeros(Nr)  # u at t=-dt
    u[:, 1] = -at[0] * dt**2 / 2  # u at t=0
    
    # Main integration loop
    for i in range(Nr):
        # Skip very short periods that might cause numerical issues
        if Tr[i] < dt*2:
            continue
            
        for j in range(1, NT):
            # Central difference update formula (converted from MATLAB)
            u[i, j+1] = (-(at[j]) + A[i] * u[i, j] + B[i] * u[i, j-1]) / C[i]
    
    # Calculate response spectra
    Sd = np.max(np.abs(u), axis=1)
    Sv = wn * Sd
    Sa = wn * Sv / 980  # Convert to g
    
    return Sd, Sv, Sa

def compute_t_path(R, rpathdur, pathdur, durslope):
    """
    Computes the path duration (Tpath) based on the hinge distances, durations, and slope.
    
    Parameters:
    R (float): Distance to the site (km).
    rpathdur (list): Hinge distances [km] (e.g., [0.0, 10.0]).
    pathdur (list): Corresponding durations at hinge distances [s] (e.g., [0.00, 0.13]).
    durslope (float): Slope for extrapolation beyond hinge distances.
    
    Returns:
    float: Path duration (Tpath) in seconds.
    """
    if R <= rpathdur[0]:
        # Below the first hinge point
        Tpath = pathdur[0]
    elif R <= rpathdur[1]:
        # Between the first and second hinge points
        Tpath = pathdur[0] + (pathdur[1] - pathdur[0]) * (R - rpathdur[0]) / (rpathdur[1] - rpathdur[0])
    else:
        # Beyond the last hinge point
        Tpath = pathdur[1] + durslope * (R - rpathdur[1])
    return Tpath


def compute_stochastic_wave(
    NS: int,
    dt: float,
    roll: float,
    beta: float,
    M: float,
    R_subfault: float,
    params: StochasticModelParameters,
    C,
    F0_main: Optional[float] = None,
    NT_main: Optional[int] = None,
    fc_subfault: Optional[float] = None,
    subdur: Optional[float] = None,
    subM0: Optional[float] = None,
    npadl: int = 0,
    n_subs: Optional[int] = None
) -> Tuple[float, np.ndarray]:
    """
    Modified compute_stochastic_wave function using parameterized approach for ground motion model.
    
    Args:
        NS (int): Number of simulations.
        dt (float): Time step (seconds).
        roll (float): Density (g/cm³).
        beta (float): Shear-wave velocity (km/s).
        M (float): Moment magnitude.
        R_subfault (float): Hypocentral distance to subfault (km).
        params (StochasticModelParameters): Model parameters.
        F0_main (float, optional): Main corner frequency (Hz).
        NT_main (int, optional): Original number of time steps.
        fc_subfault (float, optional): Corner frequency of the subfault (Hz).
        subdur (float, optional): Duration of the subfault (seconds).
        subM0 (float, optional): Seismic moment of subfault (dyne-cm).
        npadl (int, optional): Left padding samples.
        n_subs (int, optional): Number of subfaults.

    Returns:
        tuple: (mean_pga, at_padded)
            mean_pga (float): Mean Peak Ground Acceleration (g).
            at_padded (numpy.ndarray): Simulated acceleration time series (m/s²).
    """
    # Default duration if not specified (identical to original)
    if subdur is None:
        fa = 10**(2.41 - 0.533 * M)
        et = 10**(2.52 - 0.637 * M)
        fc = F0_main if F0_main is not None else fc_subfault
        fb = np.sqrt((fc**2 - (1 - et) * fa**2) / et) if fc is not None else fa
        subdur = 1 / (2 * fa) + 1 / (2 * fb) + 0.05 * R_subfault
    
    # Default seismic moment if not specified (identical to original)
    if subM0 is None:
        subM0 = 10**(1.5 * M + 16.05)
    
    # Default corner frequency if not specified (identical to original)
    if fc_subfault is None:
        sigma_bars = params.stress_drop
        fc_subfault = 4.906e6 * beta * (sigma_bars / subM0)**(1/3)
    
    # Time parameters - identical to original approach
    nseed = round(np.log2(subdur / dt)) + 1
    subN = 2**nseed if NT_main is None else NT_main
    taper = 0.05
    tmax = subN * dt
    subdf = 1 / tmax
    subNf = subN // 2
    subf = np.arange(1, subNf + 1) * subdf
    ndur = round(subdur / dt)
    ntaper = round(taper * ndur)
    nstop = ndur + 2 * ntaper
    
    # Initialize arrays for all simulations
    all_seismograms = np.zeros((NS, subN))
    
    # Loop through each simulation
    for sim in range(NS):
        # Gaussian white noise (equivalent to np.random.randn)
        nt0 = np.random.normal(0, 1, nstop)
        
        # Sargoni & Hart window function parameters - identical to original
        eps = 0.2
        eta = 0.2
        nstart = 1
        
        b = -eps * np.log10(eta) / (1 + eps * (np.log10(eps) - 1))
        c = b / (eps * subdur)
        a = (np.e / (eps * subdur))**b
        
        # Initialize arrays for window function
        win = np.zeros(nstop)
        twin = np.zeros(nstop)
        twin1 = np.zeros(nstop)
        wf = np.zeros(nstop)
        st = np.zeros(subN)
        
        # Apply window function to noise - identical to original
        for k in range(1, nstop + 1):
            k_idx = k - 1  # Adjust for 0-based indexing
            if k < nstart or k > nstop:
                win[k_idx] = 1
            else:
                if k > (nstart + ntaper) and k < (nstop - ntaper):
                    twin[k_idx] = (subdur / (nstop - nstart - 2 * ntaper)) * (k - nstart - ntaper + 1)
                    win[k_idx] = a * (twin[k_idx]**b) * np.exp(-c * twin[k_idx])
                else:
                    if k <= (nstart + ntaper):
                        twin1[k_idx] = subdur / (nstop - nstart - 2 * ntaper)
                        wf[k_idx] = a * (twin1[k_idx])**b * np.exp(-c * twin1[k_idx])
                        win[k_idx] = abs(np.sin((k - nstart) / ntaper * np.pi / 2)) * wf[k_idx]
                    else:
                        if k >= (nstop - ntaper):
                            wf[k_idx] = a * (subdur)**b * np.exp(-c * subdur)
                            win[k_idx] = abs(np.sin((nstop - k) / ntaper * np.pi / 2)) * wf[k_idx]
            
            st_idx = k_idx + npadl
            if st_idx < subN:
                st[st_idx] = win[k_idx] * nt0[k_idx]
        
        # Fourier transform - identical to original
        As = fft(st)
        Angle = np.angle(As[:subNf])
        Asf = np.abs(As[:subNf])
        Asf = Asf * dt
        
        # Normalize to unit amplitude - identical to original
        Asfsum = np.sum(Asf**2)
        AveAsf = np.sqrt(Asfsum / subNf)
        Adj = Asf / AveAsf
        
        # THIS IS WHERE WE USE THE PARAMETERIZED APPROACH INSTEAD OF TEA24
        # -------------------------------------------------------------------
        # Compute source spectrum (omega-squared model)
        #S = (2 * np.pi * subf)**2 * (subM0 / (1 + (subf / fc_subfault)**2))
        S = C * subM0 / (1 + (subf / fc_subfault)**2)
        S = (2 * np.pi * subf)**2 * S


        # Get geometric spreading
        G = params.geometric_spreading.get_spreading(R_subfault)
        
        # Get anelastic attenuation
        Q = params.quality_factor.get_Q(subf)
        An = np.exp(-np.pi * subf * R_subfault / Q)
        
        # Get high-frequency diminution
        P = params.site_attenuation.get_diminution(subf)
        
        # Get site amplification (if available)
        if hasattr(params, 'vs30'):
            vv = amfBJ(subf, beta, roll, Vs30=params.vs30)
        else:
            vv = 1.0
            
        # Combine all factors
        Ax = S * G * An * P * vv
        # -------------------------------------------------------------------
        
        # Apply a low-cut filter (high-pass) - identical to original
        fcut = 0.05
        norder = 8
        blcf = 1 / (1 + (fcut / subf)**(2 * norder))
        Ax = blcf * Ax
        
        # Apply scaling for finite fault effects - identical to original
        if F0_main is not None and n_subs is not None:
            NF = n_subs
            ScH = np.sqrt(NF) * (F0_main / fc_subfault)**2
            # Low frequency scaling
            Csc = np.sqrt(NF) / ScH
            f0eff = fc_subfault / np.sqrt(Csc)
            L1 = 1 + (subf / fc_subfault)**2
            L2 = 1 + (subf / f0eff)**2
            ScL = Csc * L1 / L2
            
            # Total scaling factor
            Sc = ScL * ScH
            Ax = Ax * Sc
        
        # Combine normalized noise and target spectrum - identical to original
        Aaf = Adj * Ax
        
        # Inverse Fourier transform with proper symmetry - identical to original
        complex_spectrum = np.zeros(subN, dtype=complex)
        complex_spectrum[:subNf] = Aaf * np.exp(1j * Angle)
        
        # Handle Nyquist frequency separately - identical to original
        if subN % 2 == 0:  # Even number of points
            complex_spectrum[subNf] = 0  # Set Nyquist frequency to 0
            complex_spectrum[subNf+1:] = np.conj(complex_spectrum[1:subNf][::-1])
        else:  # Odd number of points
            complex_spectrum[subNf:] = np.conj(complex_spectrum[1:subNf+1][::-1])
        
        # Inverse FFT and scaling - identical to original
        subAt0 = ifft(complex_spectrum) * subN
        subAt = np.real(subAt0) * 2 * subdf
        
        # Apply baseline correction - identical to original
        subAt = apply_baseline_correction(subAt, dt, 1)
        
        # Store the result
        all_seismograms[sim, :] = subAt
    
    # Define padding parameters - identical to original
    tpad = 1.0  # Padding duration in seconds
    npad = int(tpad / dt)  # Number of padding samples
    N = subN + 2 * npad  # Total length after padding
    
    # Apply pre- and post-padding - identical to original
    at_padded = np.zeros((NS, N))
    for m in range(NS):
        # Pre-pad with npad zeros
        at_pre_padded = np.pad(all_seismograms[m, :], (npad, 0), mode='constant', constant_values=0)
        # Post-pad with npad zeros
        at_padded[m, :] = np.pad(at_pre_padded, (0, npad), mode='constant', constant_values=0)
    
    # Create the filter - identical to original
    filterts = np.zeros(N)
    filterts[:npad + subN] = 1  # 1 for pre-padding and original signal, 0 for post-padding
    
    # Apply filter - identical to original
    for m in range(NS):
        at_padded[m, :] *= filterts
    
    # Adjust to NT_main if necessary - identical to original
    current_length = at_padded.shape[1]
    if NT_main is not None:
        if current_length < NT_main:
            padding_size = NT_main - current_length
            at_final = np.pad(at_padded, ((0, 0), (0, padding_size)), mode='constant', constant_values=0)
        elif current_length > NT_main:
            at_final = at_padded[:, :NT_main]
        else:
            at_final = at_padded
    else:
        at_final = at_padded
    
    # Calculate PGA in g - identical to original
    PGA_ss = np.max(np.abs(at_final), axis=1) / 9.8  # Convert m/s² to g
    mean_pga = np.mean(PGA_ss)
    
    # Return results
    return mean_pga, at_final


def apply_baseline_correction(acc, dt, BLfactor=1):
    """
    Apply baseline correction to acceleration time series.
    
    Args:
        acc (numpy.ndarray): Acceleration time series.
        dt (float): Time step (seconds).
        BLfactor (int, optional): Correction factor. Defaults to 1.
        
    Returns:
        numpy.ndarray: Baseline-corrected acceleration time series.
    """
    if BLfactor == 0:
        return acc
    
    N = len(acc)
    vel = np.zeros(N)
    
    # Calculate velocity by integrating acceleration
    for i in range(1, N):
        vel[i] = vel[i-1] + acc[i] * dt
    
    # Apply baseline correction
    t = np.arange(0, N) * dt
    a = np.polyfit(t, vel, BLfactor)
    
    # Calculate correction term
    vel_corr = np.zeros(N)
    for i in range(BLfactor + 1):
        vel_corr += a[i] * t**(BLfactor - i)
    
    # Calculate acceleration correction
    acc_corr = np.zeros(N)
    acc_corr[1:] = (vel_corr[1:] - vel_corr[:-1]) / dt
    
    # Apply correction
    acc_corrected = acc - acc_corr
    
    return acc_corrected

def mavro_papa_pulse(acceleration, dt, magnitude, 
                     gamma=0.5, nu=0.0, t0=None, peak_factor=1.0):
    """
    Apply the Mavroeidis and Papageorgiou (2003) pulse model to a ground motion time series.
    
    Parameters:
    -----------
    acceleration : numpy.ndarray
        Input acceleration time series
    dt : float
        Time step (seconds)
    magnitude : float
        Earthquake magnitude (used for period scaling)
    gamma : float, optional
        Parameter controlling the oscillatory character (default: 0.5)
    nu : float, optional
        Phase of the harmonic (default: 0.0)
    t0 : float, optional
        Time shift parameter (seconds). If None, automatically determined.
    peak_factor : float, optional
        Factor to control the pulse amplitude (default: 1.0)
        
    Returns:
    --------
    numpy.ndarray
        Modified acceleration time series with pulse component
    """
    # Calculate pulse period based on magnitude (Mavroeidis & Papageorgiou, 2003)
    pulse_period = 10**(-2.9 + 0.5 * magnitude)  # in seconds
    pulse_freq = 1.0 / pulse_period
    
    # Find the peak value of the input motion (to scale the pulse)
    peak_value = np.max(np.abs(acceleration))
    
    # Number of points
    n_points = len(acceleration)
    
    # Time vector
    time = np.arange(0, n_points) * dt
    
    # Set t0 to 1/4 of the total duration if not specified
    if t0 is None:
        t0 = 0.25 * n_points * dt
    
    # Create the pulse in time domain
    pulse = np.zeros(n_points)
    
    # PI constant
    pi = np.pi
    
    # Calculate duration of the pulse
    pulse_duration = 2.0 * gamma / pulse_freq
    
    # Filter the time points where the pulse will be applied
    t_pulse_indices = np.where((time >= t0 - pulse_duration) & (time <= t0 + pulse_duration))[0]
    
    # Apply the pulse only within the relevant time window
    for i in t_pulse_indices:
        t = time[i]
        
        # Normalized time
        t_norm = (t - t0) * pulse_freq
        
        if np.abs(t_norm) <= gamma:
            # Velocity pulse component (this is the key equation from the paper)
            pulse[i] = 0.5 * peak_value * peak_factor * (
                1.0 + np.cos(pi * t_norm / gamma)
            ) * np.cos(2.0 * pi * t_norm + nu)
    
    # Convert velocity pulse to acceleration by differentiation
    acc_pulse = np.zeros(n_points)
    acc_pulse[1:-1] = (pulse[2:] - pulse[:-2]) / (2.0 * dt)
    
    # Add the pulse to the original acceleration
    modified_acc = acceleration + acc_pulse
    
    return modified_acc



# 7. Compute Tapering Factor ( Newly added)

def simulate_subfault_wave(simulation_params, earthquake_params, R_subfault, 
                          subfault_M0, f0, C, F0_main, dur_sub, NT, npadl, 
                          scaling_factor=None, n_subs=None):
    """Simulates the wave contribution from a single subfault using the parameterized approach."""
    
    # Check if a ground motion model is provided
    if hasattr(simulation_params, 'gm_model') and simulation_params.gm_model is not None:
        # Use the ground motion model from simulation parameters
        stochastic_params = simulation_params.gm_model
        
        # Ensure stress_drop is set correctly (from earthquake_params.sigma)
        stochastic_params.stress_drop = earthquake_params.sigma * 10  # Convert MPa to bars
    else:
        # Create a new StochasticModelParameters with default values
        stochastic_params = StochasticModelParameters(
            dt=simulation_params.dt,
            ns=simulation_params.NS,
            roll=simulation_params.roll,
            beta=simulation_params.beta,
            vs30=simulation_params.Vs30 * 1000 if simulation_params.Vs30 < 10 else simulation_params.Vs30,
            stress_drop=earthquake_params.sigma * 10,  # Convert MPa to bars
            
            # Default Southwest Iberia inland parameters if no model provided
            geometric_spreading=GeometricSpreadingParameters(
                r_ref=1.0,
                segments=[(70.0, -1.1), (100.0, 0.2), (float('inf'), -1.55)]
            ),
            
            quality_factor=QualityFactorParameters(
                Q0=120.0,
                eta=0.93,
                Qmin=600.0
            ),
            
            path_duration=PathDurationParameters(
                duration_points=[(0.0, 0.0), (10.0, 0.13)],
                slope_beyond_last=0.13
            ),
            
            site_attenuation=SiteAttenuationParameters(
                kappa=simulation_params.kappa
            )
        )
    
    # Call the new function with the prepared parameters
    return compute_stochastic_wave(
        simulation_params.NS, simulation_params.dt, simulation_params.roll, 
        simulation_params.beta, earthquake_params.M, R_subfault, 
        stochastic_params, C, F0_main=F0_main, NT_main=NT, 
        fc_subfault=f0, subdur=dur_sub, subM0=subfault_M0, npadl=npadl,
        n_subs=n_subs
    )
# 12. Finalize Results
def finalize_results(all_PGA: np.ndarray) -> Tuple[float, float]:
    """Computes geometric and arithmetic means of PGA values."""
    all_PGA_np = np.array(all_PGA) #enforce to make consistent
    all_PGA_non_zero = all_PGA_np[all_PGA_np > 0]  # Filter values > 0
    if all_PGA_non_zero.size > 0:
        mean_PGA_geometric = np.exp(np.mean(np.log(all_PGA_non_zero)))
    else:
        mean_PGA_geometric = 0.0  # Or some other appropriate default value

    mean_PGA_arithmetic = np.mean(all_PGA_np)
    return mean_PGA_geometric, mean_PGA_arithmetic

import os

def export_acc_to_txt(
    output_dir: str,
    site_idx: int,
    hypo_idx: int,
    earthquake_params: EarthquakeParameters,
    site_params: SiteParameters,
    fault_params: FaultParameters,
    simulation_params: SimulationParameters,
    time_vector: np.ndarray,
    acc_data: np.ndarray,
    pga: float,
    R: float,
    sim_idx: int  # Add sim_idx as a parameter

) -> None:
    """
    Exports acceleration time series to a text file for a given site and hypocenter.

    Parameters:
    - output_dir: Directory where the text file will be saved.
    - site_idx: Index of the site (1-based in filename for readability).
    - hypo_idx: Index of the hypocenter (1-based in filename).
    - earthquake_params: Earthquake parameters (magnitude, etc.).
    - site_params: Site parameters (latitude, longitude).
    - fault_params: Fault parameters (rupture coordinates).
    - simulation_params: Simulation parameters (dt, etc.).
    - time_vector: Array of time points (s).
    - acc_data: Acceleration time series (m/s²).
    - pga: Peak Ground Acceleration (g).
    - R: Distance to fault (km).
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define filename
    #txt_filename = os.path.join(output_dir, f'acc_site_{site_idx + 1}_hypo_{hypo_idx + 1}.txt')
    txt_filename = os.path.join(output_dir, f'acc_site_{site_idx + 1}_hypo_{hypo_idx + 1}_{sim_idx + 1}.txt')

    # Open file and write metadata
    with open(txt_filename, 'w') as f:
        f.write(f'Earthquake Magnitude: {earthquake_params.M}\n')
        f.write(f'Rupture Coordinates: Latitude = {fault_params.rupture_lat}, Longitude = {fault_params.rupture_lon}\n')
        f.write(f'Site Coordinates: Latitude = {site_params.site_lat}, Longitude = {site_params.site_lon}\n')
        f.write(f'Distance to Fault (km): {R:.2f}\n')
        f.write(f'Time Step (dt): {simulation_params.dt:.5f} s\n')
        f.write(f'Peak Ground Acceleration (PGA): {pga:.6f} g\n')
        f.write('Time (s)\tAcceleration (m/s^2)\n')  # Assuming m/s²; adjust if cm/s²

        # Write time and acceleration data
        for t, acc in zip(time_vector, acc_data):
            f.write(f'{t:.5f}\t{acc:.6f}\n')

    print(f"Saved acceleration data to {txt_filename}")


def finite_fault_sim(Nhyp: int, earthquake_params: EarthquakeParameters, simulation_params: SimulationParameters,
                     site_params: SiteParameters, fault_params: FaultParameters,
                     plot_results: bool = False, output_dir: str = "acceleration_files",
                     site_idx: int = 0, calc_rsa: bool = False) -> Tuple[float, float, np.ndarray, Dict[str, Dict[str, Any]], np.ndarray, Optional[Dict]]:
    """
    Finite-fault stochastic simulation function with properly independent simulations
    """
    # Seed the random number generator for reproducibility
    np.random.seed(42)

    # 1️ Initialize fault grid (as before)
    fault_width, fault_length, num_subfaults_length, num_subfaults_width, subRadius = initialize_fault_grid(
        earthquake_params, fault_params
    )

    # 2️ Generate multiple hypocenters (as before)
    hypocenters = generate_hypocenters(
        Nhyp, fault_params, earthquake_params,
        fault_width, fault_length, num_subfaults_length, num_subfaults_width, site_params
    )

    # 3️ Compute seismic moment (as before)
    M0, averageMoment, fc_static, fc_st = compute_seismic_moment(
        earthquake_params, simulation_params, num_subfaults_length * num_subfaults_width
    )

    # Initialize rpathdur, pathdur, durslope (assuming these are constant)
    if hasattr(simulation_params, 'gm_model') and simulation_params.gm_model is not None:
        # Extract path duration parameters from the model
        duration_points = simulation_params.gm_model.path_duration.duration_points
        durslope = simulation_params.gm_model.path_duration.slope_beyond_last
        
        # Convert to separate arrays for distances and durations
        rpathdur = [point[0] for point in duration_points]
        pathdur = [point[1] for point in duration_points]
        
        print(f"Using parameterized path duration model: {len(rpathdur)} points, slope={durslope}")
    else:
        # Fallback to default values if no model is provided
        rpathdur = [0.0, 10.0]  # Default hinge distances (km)
        pathdur = [0.00, 0.13]  # Default path durations (seconds)
        durslope = 0.130  # Default slope
        print("WARNING: Using default path duration parameters")

    # 5️ Loop over hypocenters and process subfaults
    all_PGA = []
    all_At = []
    all_At_dict = {}

    # Initialize RSA results dictionary if needed
    rsa_results = None
    if calc_rsa:
        rsa_results = {
            "periods": simulation_params.Tr,
            "all_rsa": [],    # Store RSA for all hypocenters and simulations
            "mean_rsa": None  # Will store the mean RSA at the end
        }


    for idx, hypo_data in enumerate(hypocenters, 1):
        print(f"Processing Hypocenter {idx} at i0={hypo_data['i0']}, j0={hypo_data['j0']}")

        #  Initialize i0, j0 properly
        i0, j0 = hypo_data["i0"], hypo_data["j0"]
        R_hypo = hypo_data["R"]  # Call parameter from generate_hypocenter

        # Calculate number of effective subfaults
        no_effective_subfaults = max(1, round(num_subfaults_length * (simulation_params.pulsing_percent / 100.0) / 2))

        #  Process each subfault within the fault grid
        #  Compute Epicentral Distance and Azimuth
        _, _, R_epicentral, Az = compute_site_location(1, site_params.site_lat, site_params.site_lon,
                                                         fault_params.rupture_lat, fault_params.rupture_lon)
        #  Initialize rupture variables for each hypocenter
        rupture_parameters = RuptureParameters(R_hypo, earthquake_params.h_ref, earthquake_params.dip,
                                               earthquake_params.strike, Az)  # creating object

        # Call process_subfaults *before* the nested loops
        active_subfaults, R_subfaults, delay, t_arrive, rise_time, dur_path, dur_sub, t_arrive_min, t_end_max = process_subfaults(
                num_subfaults_length, num_subfaults_width, i0, j0, fault_params.subfault_size,
                fault_params.subfault_size,
                simulation_params, simulation_params.beta, subRadius, rpathdur, pathdur, durslope,
                no_effective_subfaults, rupture_parameters  
            )

        # 4️ Initialize time and frequency parameters - AFTER subfault processing
        t1, NT, df, fmax, C, M0 = initialize_time_frequency(simulation_params, earthquake_params, rise_time,
                                                    t_arrive, dur_sub)

        #  Calculate slip weights and moments
        slip_weights = np.zeros((num_subfaults_length, num_subfaults_width))
        for i in range(num_subfaults_length):
            for j in range(num_subfaults_width):
                slip_weights[i, j] = np.random.rand()

        total_weight = np.sum(slip_weights)
        subfault_moments = M0 * slip_weights / total_weight

        # Calculate dynamic corner frequency based on number of active subfaults
        dynamic_corner_frequencies = np.zeros((num_subfaults_length, num_subfaults_width))

        # Active Subfaults Logic
        NR = np.zeros((num_subfaults_length, num_subfaults_width))
        for i in range(num_subfaults_length):
            for j in range(num_subfaults_width):
                NR[i, j] = number_of_pulsing_subs(i + 1, j + 1, i0 + 1, j0 + 1, num_subfaults_length,
                                                  num_subfaults_width, no_effective_subfaults)
                if NR[i, j] == 0:
                    NR[i, j] = 1  # Ensure minimum value of 1

        # Initialize other parameters
        subf0 = np.zeros((num_subfaults_length, num_subfaults_width))
        subTrise = np.zeros((num_subfaults_length, num_subfaults_width))
        subTpath = np.zeros((num_subfaults_length, num_subfaults_width))
        subR = np.zeros((num_subfaults_length, num_subfaults_width))
        subdur = np.zeros((num_subfaults_length, num_subfaults_width))
        subdelay = np.zeros((num_subfaults_length, num_subfaults_width))
        subTarrive = np.zeros((num_subfaults_length, num_subfaults_width))
        subTend = np.zeros((num_subfaults_length, num_subfaults_width))

        # Subf0 calculations
        firstf0 = fc_st
        for i in range(num_subfaults_length):
            for j in range(num_subfaults_width):
                subf0[i, j] = firstf0 * NR[i, j] ** (-1 / 3)
                # Correct the subfault size
                subR[i, j] = compute_subfault_distance(R_epicentral, earthquake_params.h_ref,
                                                        earthquake_params.dip, earthquake_params.strike, Az,
                                                        fault_params.subfault_size, fault_params.subfault_size, i + 1,
                                                        j + 1)

                subTpath[i, j] = compute_t_path(subR[i, j], rpathdur, pathdur, durslope)
                subTrise[i, j] = subRadius / simulation_params.rupture_velocity

                subdur[i, j] = subTrise[i, j] + subTpath[i, j]

                subdelay[i, j] = np.sqrt(
                    (fault_params.subfault_size * (i - i0)) ** 2 + (
                                fault_params.subfault_size * (j - j0)) ** 2) / simulation_params.rupture_velocity if (
                            i != i0 or j != j0) else 0.0
                subTarrive[i, j] = subdelay[i, j] + subR[i, j] / simulation_params.beta

                subTend[i, j] = subTarrive[i, j] + subdur[i, j]

        stutter = np.zeros((num_subfaults_length, num_subfaults_width))

        # Generate random stutter delays for complexity in the rupture process
        for i in range(num_subfaults_length):
            for j in range(num_subfaults_width):
                stutter[i, j] = np.random.rand() * subTrise[i, j]

        t_arrive_min = np.min(subTarrive)

        nshift = np.zeros((num_subfaults_length, num_subfaults_width))
        # Check conditionals
        if (np.isnan(subTarrive).any() == False) and (np.isnan(t_arrive_min) == False) and (
        np.isnan(stutter).any() == False):
            nshift = np.round((subTarrive - t_arrive_min + stutter) / simulation_params.dt).astype(int)
        else:
            nshift = np.zeros_like(subTarrive)

        npadl = int(10/simulation_params.dt)
        df = 1/(NT * simulation_params.dt)
        f_array = np.arange(0, fmax, df)
        n_subs = num_subfaults_length * num_subfaults_width

        # NEW APPROACH: Process each simulation as an independent entity
        # Store all simulation results
        all_sim_accel = []
        all_sim_pga = []

        # Loop over all simulations
        for sim_idx in range(simulation_params.NS):
            # Set a unique random seed for this simulation
            np.random.seed(42 + sim_idx)
            
            # Initialize acceleration array for this simulation
            acc_total_sim = np.zeros(NT)
            
            # Process all subfaults for this single simulation
            for i in range(num_subfaults_length):
                for j in range(num_subfaults_width):
                    # Only process active subfaults
                    if active_subfaults[i, j] == 1:
                        # Calculate parameters for this subfault
                        f0 = subf0[i, j]
                        subfault_M0 = subfault_moments[i, j]
                        
                        # Generate a SINGLE subfault contribution
                        _, acc_single = simulate_subfault_wave_single(
                            simulation_params, earthquake_params, R_subfaults[i, j],
                            subfault_M0, f0, C, fc_static,
                            dur_sub[i, j], NT, npadl,
                            scaling_factor=None, n_subs=n_subs
                        )
                        
                        # Ensure acc_single is the right length
                        if len(acc_single) > NT:
                            acc_single = acc_single[:NT]
                        elif len(acc_single) < NT:
                            acc_single = np.pad(acc_single, (0, NT - len(acc_single)), mode="constant")
                        
                        # Apply shift and add to total
                        delay_steps = nshift[i, j]
                        acc_total_sim += np.roll(acc_single, delay_steps)
            
            # Apply pulse model if enabled
            if simulation_params.pulse_params.enabled:
                acc_total_sim = mavro_papa_pulse(
                    acc_total_sim,
                    simulation_params.dt,
                    earthquake_params.M,
                    gamma=simulation_params.pulse_params.gamma,
                    nu=simulation_params.pulse_params.nu,
                    t0=simulation_params.pulse_params.t0,
                    peak_factor=simulation_params.pulse_params.peak_factor
                )
            
            # Store this simulation's complete time series
            all_sim_accel.append(acc_total_sim)
            
            # Calculate PGA for this complete simulation
            pga_sim = np.max(np.abs(acc_total_sim)) / 981  # Convert to g
            all_sim_pga.append(pga_sim)
            
            # Export acceleration data to text file for each simulation
            export_acc_to_txt(
                output_dir=output_dir,
                site_idx=site_idx,
                hypo_idx=idx - 1,  # Adjust to 0-based index
                earthquake_params=earthquake_params,
                site_params=site_params,
                fault_params=fault_params,
                simulation_params=simulation_params,
                time_vector=t1,
                acc_data=acc_total_sim,
                pga=pga_sim,
                R=R_epicentral,
                sim_idx=sim_idx
            )

        # Convert lists to arrays
        all_sim_accel = np.array(all_sim_accel)
        all_sim_pga = np.array(all_sim_pga)
        
        # Store in output structures
        At_hypo = all_sim_accel
        PGA_hypo = all_sim_pga
        all_PGA.append(PGA_hypo)
        
        # Calculate RSA for each simulation if requested
        if calc_rsa:
            hypo_rsa = np.zeros((simulation_params.NS, len(simulation_params.Tr)))
            for sim_idx in range(simulation_params.NS):
                _, _, rsa_sim = calc_psa_cdm(all_sim_accel[sim_idx, :], simulation_params.dt, simulation_params.Tr)
                hypo_rsa[sim_idx, :] = rsa_sim
            
            # Store RSA for this hypocenter in the results dictionary
            rsa_results[f"Hypocenter_{idx}_RSA"] = hypo_rsa
            rsa_results["all_rsa"].append(hypo_rsa)  # Store for averaging later

        # Add hypocenter results to dictionary
        all_At_dict[f"Hypocenter_{idx}"] = {
            "Acceleration": At_hypo,
            "Time(s)": t1
        }
        
        # Print some statistics
        print(f"Mean PGA: {np.mean(PGA_hypo):.4f} g")

    # Compute final PGA results
    mean_PGA_geometric, mean_PGA_arithmetic = finalize_results(np.array(all_PGA))
    
    # Compute average RSA across all hypocenters and simulations
    if calc_rsa:
        # Stack all RSA arrays
        all_rsa_stack = np.vstack([rsa for hypo_rsa in rsa_results["all_rsa"] for rsa in hypo_rsa])
        
        # Calculate mean across all simulations and hypocenters
        rsa_results["mean_rsa"] = np.mean(all_rsa_stack, axis=0)

    # Plot results if requested
    if plot_results:
        plot_idx = 0 
        num_plots = max(1, len(all_At_dict))
        num_rows = num_plots
        num_cols = 1
    
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows), dpi=100)
    
        # Ensure axes is a NumPy array, even if only one plot
        if num_plots == 1:
            axes = np.array([axes])
        axes = axes.reshape(num_rows, num_cols)
        
        # Plot each hypocenter's first simulation
        for idx, (hypo_name, hypo_data) in enumerate(list(all_At_dict.items())[:num_plots]):
            if hypo_name == "Uniform_Time(s)":
                continue
    
            # Extract acceleration time series and time array for the current hypocenter
            At_hypo = hypo_data["Acceleration"]
            t1 = hypo_data["Time(s)"]
    
            # Calculate Epicentral Distance and Azimuth
            _, _, R_epicentral, _ = compute_site_location(1, site_params.site_lat, site_params.site_lon, 
                                                        fault_params.rupture_lat, fault_params.rupture_lon)
    
            # Plot ONLY the first simulation
            sim = 0  # Index of the first simulation
    
            axes[idx, 0].plot(
                t1,
                At_hypo[sim, :],
                label=f"Total Acceleration - Simulation {sim + 1}",
                alpha=0.7,
                color='blue'
            )
    
            # Add title, labels, and legend
            title = f"Total Accel. - {hypo_name}\nR = {R_epicentral:.2f} km"
            axes[idx, 0].set_title(title)
            axes[idx, 0].set_xlabel("Time (s)")
            axes[idx, 0].set_ylabel("Acceleration (cm/s²)")
            axes[idx, 0].legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., ncol=1)
            axes[idx, 0].grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.show()

    if calc_rsa:
        return mean_PGA_geometric, mean_PGA_arithmetic, t1, all_At_dict, np.array(all_PGA), rsa_results
    else:
        return mean_PGA_geometric, mean_PGA_arithmetic, t1, all_At_dict, np.array(all_PGA), None


def simulate_subfault_wave_single(simulation_params, earthquake_params, R_subfault, 
                               subfault_M0, f0, C, F0_main, dur_sub, NT, npadl, 
                               scaling_factor=None, n_subs=None):
    """
    Simulates a single subfault wave contribution (for a single simulation).
    This is a modified version of simulate_subfault_wave that returns just one time series.
    
    Returns:
        tuple: (mean_pga, accel_single)
            mean_pga (float): Peak Ground Acceleration (g).
            accel_single (numpy.ndarray): Single acceleration time series.
    """
    # Check if a ground motion model is provided
    if hasattr(simulation_params, 'gm_model') and simulation_params.gm_model is not None:
        # Use the ground motion model from simulation parameters
        stochastic_params = simulation_params.gm_model
        
        # Ensure stress_drop is set correctly (from earthquake_params.sigma)
        stochastic_params.stress_drop = earthquake_params.sigma * 10  # Convert MPa to bars
    else:
        # Create a new StochasticModelParameters with default values
        stochastic_params = StochasticModelParameters(
            dt=simulation_params.dt,
            ns=1,  # Always use 1 here since we want a single realization
            roll=simulation_params.roll,
            beta=simulation_params.beta,
            vs30=simulation_params.Vs30  if simulation_params.Vs30 < 10 else simulation_params.Vs30,
            stress_drop=earthquake_params.sigma,  # Convert MPa to bars
            
            # Default Southwest Iberia inland parameters if no model provided
            geometric_spreading=GeometricSpreadingParameters(
                r_ref=1.0,
                segments=[(70.0, -1.1), (100.0, 0.2), (float('inf'), -1.55)]
            ),
            
            quality_factor=QualityFactorParameters(
                Q0=120.0,
                eta=0.93,
                Qmin=600.0
            ),
            
            path_duration=PathDurationParameters(
                duration_points=[(0.0, 0.0), (10.0, 0.13)],
                slope_beyond_last=0.13
            ),
            
            site_attenuation=SiteAttenuationParameters(
                kappa=simulation_params.kappa
            )
        )
    
    # Call compute_stochastic_wave with NS=1 to get a single time series
    pga, acc_array = compute_stochastic_wave(
        1, simulation_params.dt, simulation_params.roll, 
        simulation_params.beta, earthquake_params.M, R_subfault, 
        stochastic_params, C, F0_main=F0_main, NT_main=NT, 
        fc_subfault=f0, subdur=dur_sub, subM0=subfault_M0, npadl=npadl,
        n_subs=n_subs
    )
    
    # Extract the single time series
    return pga, acc_array[0, :]