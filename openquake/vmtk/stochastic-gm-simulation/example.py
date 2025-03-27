# -*- coding: utf-8 -*-
"""
Simplified Finite Fault Simulation Example
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seismic_wave_generator

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# For cases with a larger number of sites, better to use a sites.csv file:
# data_file_path = os.path.join(current_dir, 'data', 'sites.csv')
# sites = pd.read_csv(data_file_path)

# Using manually defined site coordinates instead
site_coords = [
    {'lon': -9.5, 'lat': 38.78},
    {'lon': -9.48, 'lat': 38.76},
    {'lon': -9.48, 'lat': 38.78},
    {'lon': -9.48, 'lat': 38.8}
]
sites = pd.DataFrame(site_coords)

# Output directory for accelerograms
output_folder = os.path.join(current_dir, 'data', 'ACC')
os.makedirs(output_folder, exist_ok=True)

# Define ground motion model parameters for Southwest Iberia inland region
my_model = seismic_wave_generator.StochasticModelParameters(
    # Geometric spreading parameters
    geometric_spreading=seismic_wave_generator.GeometricSpreadingParameters(
        r_ref=1.0,
        segments=[(70.0, -1.1), (100.0, 0.2), (float('inf'), -1.55)]
    ),
    
    # Quality factor parameters
    quality_factor=seismic_wave_generator.QualityFactorParameters(
        Q0=120.0,
        eta=0.93,
        Qmin=600.0
    ),
    
    # Path duration parameters
    path_duration=seismic_wave_generator.PathDurationParameters(
        duration_points=[
            (0.0, 0.0),    
            (10.0, 0.13),
            (70.0, 0.09),
            (120.0, 0.05),
        ],
        slope_beyond_last=0.05
    ),
    
    # Site attenuation parameter
    site_attenuation=seismic_wave_generator.SiteAttenuationParameters(
        kappa=0.033
    ),
    
    # Other parameters
    stress_drop=200.0,  # in bars
    roll=2.8,  # density (g/cm³)
    beta=3.5,  # shear-wave velocity (km/s)
    vs30=760.0  # in m/s
)

# Earthquake and fault parameters
earthquake_params = seismic_wave_generator.EarthquakeParameters(
    M=6.0,
    rake=90,
    strike=30.0,
    dip=15.0,
    h_ref=5.0,
    stress_ref=70,
    sigma=200  # in bars
)

# Simulation parameters with ground motion model
simulation_params = seismic_wave_generator.SimulationParameters(
    NS=10,  # Number of simulations
    dt=0.005,  # Time step
    roll=2.8,  # Density
    beta=3.5,  # Shear-wave velocity
    Vs30=0.76,  # Site Vs30 in km/s
    Tr=np.concatenate([  # Periods for response spectra
        np.arange(0.02, 0.26, 0.05),
        np.arange(0.3, 1., 0.05),
        np.arange(1.1, 2.1, 0.15),
        np.arange(2.2, 3.1, 0.5),
        np.arange(3, 5.6, 1)
    ]),
    kappa=0.033,  # Site kappa
    tpad=20,  # Time padding
    pulsing_percent=50.0,  # Percentage of pulsing subfaults
    rupture_velocity=2.8,  # Rupture velocity
    gm_model=my_model,  # Ground motion model
    pulse_params=seismic_wave_generator.PulseParameters(  # Pulse model parameters
        enabled=False,  # Disable by default
        gamma=2.0,
        nu=0.0,
        t0=None,
        peak_factor=1.5
    )
)

# Fault parameters
fault_params = seismic_wave_generator.FaultParameters(
    subfault_size=2.0,
    rupture_lat=38.8,
    rupture_lon=-9.40,
)

# Preallocate arrays for results
num_sites = len(sites)
R = np.zeros(num_sites)
PGA_finite_fault = np.zeros(num_sites)
R_rup = np.zeros(num_sites)  
R_jb = np.zeros(num_sites)  

# Loop over each site
for site_idx, site in sites.iterrows():
    site_lat, site_lon = site['lat'], site['lon']
    print(f'Processing Site {site_idx + 1}/{num_sites} (Lat: {site_lat}, Lon: {site_lon})')

    # Compute epicentral distance and azimuth
    _, _, R_epicentral, Az = seismic_wave_generator.compute_site_location(1, site_lat, site_lon, fault_params.rupture_lat, fault_params.rupture_lon)
    
    # Compute Rhypo (hypocentral distance)
    Rhypo = seismic_wave_generator.compute_point_source_distance(R_epicentral, earthquake_params.h_ref, earthquake_params.dip, earthquake_params.strike, Az)
    R[site_idx] = Rhypo
    print(f'Site {site_idx + 1}: Rhypo = {Rhypo:.2f} km')

    # Calculate fault dimensions
    fault_width, fault_length = seismic_wave_generator.width_length(earthquake_params.M, earthquake_params.rake, earthquake_params.sigma, earthquake_params.stress_ref)
    
    # Calculate Rrup and Rjb distances
    R_rup[site_idx], R_jb[site_idx] = seismic_wave_generator.calculate_fault_distances(
        site_lat, site_lon, 
        fault_params.rupture_lat, fault_params.rupture_lon,
        earthquake_params.strike, earthquake_params.dip, 
        earthquake_params.h_ref,
        fault_width, fault_length)

    # Create a SiteParameters instance for each site
    site_params = seismic_wave_generator.SiteParameters(
        site_lat=site_lat,
        site_lon=site_lon
    )

    # Run finite-fault simulation
    Nhyp = 1  # Number of hypocenters
    PGA_ff, _, t1, all_At_dict, all_PGA_array, _  = seismic_wave_generator.finite_fault_sim(
        Nhyp,
        earthquake_params,
        simulation_params,
        site_params,
        fault_params,
        plot_results=True,  # Plot time series
        output_dir=output_folder,
        site_idx=site_idx,
        calc_rsa=False  # Disable RSA calculation for simplicity
    )

    PGA_finite_fault[site_idx] = PGA_ff
    
