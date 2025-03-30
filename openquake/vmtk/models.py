# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:39:31 2025

@author: Amir Taherian
"""


import numpy as np
import os
import pandas as pd
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


def InputFAS(f, beta, roll, subR, subM0,fc, kappa):
    """
    Compute the Fourier Amplitude Spectrum (FAS).
    
    Parameters:
        beta0 (float): Source shear wave velocity (km/s).
        roll0 (float): Source density (g/cm³).
        subf (array): Frequency array (Hz).
        subR (float): Subfault distance (km).
        C (float): Spectral constant.
        subM0 (float): Subfault moment (dyne-cm).
        subf0 (float): Corner frequency (Hz).
        kappa (float): High-frequency attenuation parameter (s).
    
    Returns:
        subFAS (array): Fourier amplitude spectrum (cm/s).
    """
    # Constant C, with unit conversion accounted for
    C = (0.55 * 2.0 * 0.707) / (4 * np.pi * (roll * 1000) * (beta * 1000)**3 * 1000)

    # Source model based on Brune's stress drop
    #fc = 4.906 * 10**6 * beta * (sigma / subM0)**(1 / 3)
    E = C * subM0 / (1 + (f / fc)**2)
    E = (2 * np.pi * f)**2 * E  # Convert source term to acceleration (dyne/cm)
    E = E / 1e7  # Convert to m/s

    # Geometric spreading
    if subR <= 50:
        G = subR ** -1.0
    elif subR <= 90:
        G = ((50 ** 0.3) / (50 ** 1.0)) * (subR ** -0.3)
    elif subR <= 120:
        G = ((50 ** 0.3) / (50 ** 1.0)) * ((90 ** 1.1) / (90 ** 0.3)) * (subR ** -1.1)
    else:
        G = ((50 ** 0.3) / (50 ** 1.0)) * ((90 ** 1.1) / (90 ** 0.3)) * ((120 ** 0.5) / (120 ** 1.1)) * (subR ** -0.5)

    # Anelastic attenuation
    cq = 3.5  # Propagation velocity in crust (km/s)
    Q0 = 180
    nq = 0.5
    Q = Q0 * (f ** nq)  # Attenuation quality factor
    Ae1 = -np.pi * f * subR
    Ae2 = Q * cq
    # Avoid division by zero or negative Q
    Ae2 = np.where(Ae2 > 0, Ae2, np.inf)  # Replace non-positive Ae2 with infinity

    Ae = np.exp(Ae1 / Ae2)  # Anelastic attenuation

    # Crustal amplification
    Am = amfBJ(f, beta, roll, Vs30=0.76)  # Replace with your amplification function
    Am = 1
    # High-frequency attenuation
    An = np.exp(-np.pi * f * kappa)

    # Final Fourier Amplitude Spectrum
    subFAS = E * Ae * Am * An * G * 100  # Unit: cm/s


    return subFAS
def AB95(f, M, R, roll, beta, sigma, Vs30, fm):
    """
    AB95 double corner frequency model (Atkinson & Boore, 1995)
    
    Parameters:
    f (array): Frequency array (Hz)
    M (float): Magnitude
    R (float): Distance (km)
    roll (float): Density (g/cm^3)
    beta (float): Shear wave velocity (km/s)
    sigma (float): Stress parameter (bars)
    Vs30 (float): Average shear-wave velocity over the top 30 m (m/s)
    fm (float): High-frequency attenuation parameter
    
    Returns:
    Ax (array): Ground motion acceleration (cm/s)
    """
    
    # The applicable range of this model is: 4.0 <= M <= 7.25, 10 <= R <= 500 km, 0.5 <= f <= 20 Hz

    # Geometric attenuation function (Trilinear form)
    if R <= 70:
        G = R**-1
    elif R <= 130:
        G = 70**-1
    else:
        G = 70**-1 * (R / 130)**-0.5

    # Source model
    fa = 10**(2.41 - 0.533 * M)
    et = 10**(2.52 - 0.637 * M)
    M0 = 10**(1.5 * M + 16.05)  # (Hanks & Kanamori, 1979; Boore et al., 2014)
    
    # Constant C, with unit conversion accounted for
    C = (0.55 * 2.0 * 0.707) / (4 * np.pi * (roll * 1000) * (beta * 1000)**3 * 1000)
    
    # Corner frequencies calculation
    fc = 4.906 * 10**6 * beta * (sigma / M0)**(1/3)
    fb = np.sqrt((fc**2 - (1 - et) * fa**2) / et)  # (Boore et al., 2014)

    # Spectral shape factors
    sa = 1 / (1 + (f / fa)**2)
    sb = 1 / (1 + (f / fb)**2)
    S = C * M0 * (sa * (1 - et) + sb * et)
    S = (2 * np.pi * f)**2 * S
    S = S / 10**7  # Convert unit for S into m/s

    # Upper-crust attenuation model
    if fm > 1:
        P = (1 + (f / fm)**8)**-0.5
    else:
        kappa = fm
        P = np.exp(-np.pi * f * kappa)

    # Anelastic whole path attenuation
    Q0 = 680
    n = 0.36

    Q = Q0 * f**n
    Q = Q * beta
    Q = Q**-1
    An = np.exp(-np.pi * f * Q * R)

    # Amplification based on Boore-Joyner (BJ) shear-wave velocity profiling model
    vv = amfBJ(f, beta, roll, Vs30)  # Assume amfBJ is defined elsewhere
    #vv=1
    # Calculate the ground motion acceleration (Ax) and convert it into cm/s
    Ax = S * An * P * vv * G * 100

    return Ax


# Adjusting the function name as requested



from scipy.interpolate import interp1d

def amfBJ(f, beta, roll, Vs30):
    """
    Amplification model function based on Boore & Joyner shear-wave velocity profiling model (BJ).

    Parameters:
    f (array): Frequency array (Hz)
    beta (float): Shear wave velocity (km/s)
    roll (float): Density (g/cm^3)
    Vs30 (float): Average shear-wave velocity over the top 30 m (m/s)

    Returns:
    vv (array): Amplification factor
    """

    # Depth values (H) in km
    H=np.array([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.012,
    0.014,0.016,0.018,0.02,0.022,0.024,0.026,0.028,0.03,0.032,0.034,0.036,
    0.038,0.04,0.042,0.044,0.046,0.048,0.05,0.052,0.054,0.056,0.058,0.06,
    0.062,0.064,0.066,0.068,0.07,0.072,0.074,0.076,0.078,0.08,0.082,0.104,
    0.126,0.147,0.175,0.202,0.23,0.257,0.289,0.321,0.353,0.385,0.42,0.455,
    0.49,0.525,0.562,0.599,0.637,0.674,0.712,0.751,0.789,0.827,0.866,0.906,
    0.945,0.984,1.02,1.06,1.1,1.14,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.5,
    4,4.5,5,5.5,6,6.5,7,7.5,8,10,15,20,30,50])

    # Shear-wave velocity for generic very hard rock (V1) in km/s
    V1=np.array([2.768,2.7688,2.7696,2.7704,2.7712,2.772,2.7728,2.7736,2.7744,2.7752,
    2.776,2.7776,2.7792,2.7808,2.7824,2.784,2.7856,2.7872,2.7888,2.7904,
    2.792,2.7936,2.7952,2.7968,2.7984,2.8,2.8016,2.8032,2.8048,2.8064,2.808,
    2.80956,2.81112,2.81268,2.81424,2.8158,2.81736,2.81892,2.82048,2.82204,
    2.8236,2.82516,2.82672,2.82828,2.82984,2.8314,2.83296,2.85004,2.86676,
    2.88272,2.9035,2.92344,2.9436,2.9629,2.9853,3.00686,3.06098,3.0821,3.0718,
    3.0941,3.1158,3.1365,3.15796,3.17942,3.20072,3.22048,3.24024,3.260835504,
    3.271637476,3.281964539,3.29211285,3.302087707,3.311425176,3.320409798,
    3.32841313,3.337002316,3.345294257,3.353309507,3.364853485,3.399786096,
    3.430339103,3.457516593,3.482010086,3.50431659,3.510651329,3.516529187,
    3.521980007,3.527062193,3.538443827,3.54833273,3.557078298,3.564919739,
    3.572028075,3.578529853,3.584521359,3.590077571,3.595258021,3.600110775,
    3.616939825,3.647720809,3.669718999,3.700949146,3.740673099])


    # Shear-wave velocity for generic rock (V2) in km/s
    V2=np.array([0.245,0.245,0.406898105,0.454341586,0.491321564,0.522065927,0.548608647,
    0.572100299,0.593261253,0.612575285,0.630384474,0.662434293,0.690800008,
    0.716351448,0.739672767,0.761177017,0.781168059,0.799876552,0.817482113,
    0.834127602,0.84992869,0.872659052,0.894459078,0.915511227,0.935880677,
    0.955623835,0.974789894,0.993422052,1.011558478,1.02923309,1.046476183,
    1.063314944,1.079773879,1.095875162,1.111638937,1.127083562,1.142225823,
    1.157081117,1.171663602,1.185986333,1.200061379,1.21389992,1.22751234,
    1.240908299,1.254096801,1.26708626,1.279884545,1.409876676,1.524401504,
    1.623105361,1.742468929,1.822101865,1.869784562,1.91154454,1.956709713,
    1.998031044,2.036174042,2.071640608,2.10782397,2.141667263,2.173485515,
    2.203532354,2.23359926,2.262120148,2.289978882,2.315853324,2.341268645,
    2.366247007,2.389604645,2.412077883,2.434298272,2.456270778,2.4769581,
    2.496972474,2.514890987,2.534215818,2.55296508,2.571175941,2.597555276,
    2.678472608,2.750601065,2.815833418,2.875495547,2.930554778,2.981739972,
    3.029614889,3.074625172,3.117129605,3.214232374,3.300788279,3.33118689,
    3.361507951,3.389174372,3.414630616,3.438216903,3.460199618,3.48079135,
    3.500164545,3.567982556,3.694592725,3.787139494,3.921526466,4.097643229])

    # Calculate S1, S2, and interpolate shear-wave velocity profile
    S1 = 1 / V1
    S2 = 1 / V2
    betavs = (1 / Vs30 - 1 / 0.618) / (1 / 2.780 - 1 / 0.618)
    Nh = len(H)
    S = np.zeros(Nh)
    for i in range(Nh):
        S[i] = betavs * S1[i] + (1 - betavs) * S2[i]

    V = 1 / S

    # Initialize arrays for various parameters
    thick = np.zeros(Nh - 1)
    wtt = np.zeros(Nh - 1)
    acct = np.zeros(Nh - 1)
    period = np.zeros(Nh - 1)
    avev = np.zeros(Nh - 1)
    fn = np.zeros(Nh - 1)

    # Compute wave travel time, accumulated time, and period for each layer
    for m in range(1, Nh):
        thick[m - 1] = H[m] - H[m - 1]
        wtt[m - 1] = thick[m - 1] / ((V[m] + V[m - 1]) / 2)
        acct[m - 1] = np.sum(wtt[:m])
        period[m - 1] = 4 * acct[m - 1]
        avev[m - 1] = (H[m] - H[0]) / acct[m - 1]
        fn[m - 1] = 1 / period[m - 1]

    # Trim the arrays properly to ensure they match in length
    fn = fn[1:]  # Skip the first frequency entry
    avev = avev[1:]  # Skip the first velocity entry

    # Calculate Density using the V values
    # Initialize arrays for Density and Vp
    Density = np.zeros(len(V))
    Vp = np.zeros(len(V))
    
    # Loop to compute Vp and Density
    for i in range(len(V)):
        if V[i] < 0.3:
            # Case for V < 0.3
            Density[i] = 1 + (1.53 * V[i]**0.85) / (0.35 + 1.889 * V[i]**1.7)
        else:
            # Case for 0.3 <= V < 3.55
            Vp[i] = 0.9409 + V[i] * 2.0947 - 0.8206 * V[i]**2 + 0.2683 * V[i]**3 - 0.0251 * V[i]**4
            if V[i] < 3.55:
                Density[i] = 1.74 * Vp[i]**0.25
            else:
                # Case for V >= 3.55
                Density[i] = (
                    1.6612 * Vp[i]
                    - 0.4721 * Vp[i]**2
                    + 0.0671 * Vp[i]**3
                    - 0.0043 * Vp[i]**4
                    + 0.000106 * Vp[i]**5
                )

    # Adjust the length of Density to match avev
    Density = Density[:len(avev)]

    # Check if lengths match after trimming
    if len(avev) != len(Density):
        raise ValueError(f"Mismatch between avev ({len(avev)}) and Density ({len(Density)}).")

    # Calculate amplification factor
    amp = np.sqrt((beta * roll) / (avev * Density))
    # Interpolate using log amplification-linear frequency interpolation
    fn = np.flip(fn)
    amp = np.flip(amp)

    # Logarithmic amplification and interpolation
    vn = np.log(amp)  # Convert amplification to log space
    interp_func = interp1d(
        fn, vn, kind="linear", fill_value="extrapolate"
    )  # Use `fill_value="extrapolate"`
    vv_log = interp_func(f)  # Apply the interpolation
    vv = np.exp(vv_log)  # Convert back to linear space


    return vv


def TEA24(f,M, R, roll, beta, sigma, kappa, C, region="inland",M0_override=None,f0_override=None):
    """
    Ground Motion Model for Southwest Iberia (renamed as TEA24 based on proposed geometrical spreading and stochastic model parameters)
    Includes source spectra logic from EXSIM and supports M0_override and fc_dynamic for finite-fault simulations.
    
    Parameters:
    f (array): Frequency array (Hz)
    M (float): Magnitude
    R (float): Distance (km)
    roll (float): Density (g/cm^3)
    beta (float): Shear wave velocity (km/s)
    sigma (float): Stress parameter (bars)
    fm (float): High-frequency attenuation parameter
    region (str): Either 'inland' or 'offshore' based on study model divisions
    M0_override (float): Optional seismic moment override (dyne·cm) for subfaults
    fc_dynamic (float): Optional dynamic corner frequency override (Hz) for subfaults
    return_tp_only (bool): If True, only returns duration `tp`
    
    Returns:
    Ax (array): Ground motion acceleration (cm/s)
    tp (float): Duration (s)
    """
    # Geometrical spreading model based on region
    if region == "inland":
        if R <= 70:
            G = R**-1.1
        elif R <= 100:
            G = 70**-1.1 * (R / 70)**0.2
        else:
            G = 70**-1.1 * (100 / 70)**0.2 * (R / 100)**-1.55
    else:  # offshore model
        if R <= 115:
            G = R**-1.1
        else:
            G = 115**-1.1 * (R / 115)**-1.5


    S = C * M0_override / (1 + (f / f0_override)**2)
    S = (2 * np.pi * f)**2 * S

    # High-frequency attenuation
    if kappa > 1:
        P = (1 + (f / kappa)**8)**-0.5
    else:
        P = np.exp(-np.pi * f * kappa)

    # Anelastic attenuation based on regional Q model
    if region == "inland":
        Q0, n = 120, 0.93
        Qmin = 600
    else:
        Q0, n = 165, 1.07
        Qmin = 800
    Q = np.maximum(Q0 * f**n, Qmin)
    An = np.exp(-np.pi * f * R / Q)

    # Geometric spreading and amplification
    vv = amfBJ(f, beta, roll, Vs30=0.76)  
    vv = 1
    Ax = S * An * P * vv * G 

    # Duration model based on region
    if region == "inland":
        if R <= 70:
            tp = 0.13 * R
        elif R <= 120:
            tp = 70 * 0.13 + (R - 70) * 0.09
        else:
            tp = 70 * 0.13 + 50 * 0.09 + (R - 120) * 0.05
    else:  # offshore model
        if R <= 115:
            tp = 0.12 * R
        else:
            tp = 115 * 0.12 + (R - 115) * 0.02
    return Ax



