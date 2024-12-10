import math 
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy import spatial, stats
from scipy.interpolate import interp1d

## Define plot style
HFONT = {'fontname':'Helvetica'}

FONTSIZE_1 = 16
FONTSIZE_2 = 14
FONTSIZE_3 = 12

LINEWIDTH_1= 3
LINEWIDTH_2= 2
LINEWIDTH_3 = 1

RESOLUTION = 500
MARKER_SIZE_1 = 100
MARKER_SIZE_2 = 60
MARKER_SIZE_3 = 10

GEM_COLORS  = ["#0A4F4E","#0A4F5E","#54D7EB","#54D6EB","#399283","#399264","#399296"]

class risk_targetor():

    """
    Details
    -------
    Class of functions to post-process results of nonlinear time-history analysis
    """
    
    def __init__(self, imls, apoes, target_probability, target_frequency, beta, pflag=True):
        """
        ----------
        Parameters
        ----------
        imls:                     list                Intensity measure levels (x-axis of the hazard curve)
        apoes:                    list                Annual probabilities of exceedance
        target_probability:      float                Target probability of failure
        target_frequency:        float                Target annual rate (hazard frequency)
        target_frequency:        float                Fragility function dispersion
        pflag:                    bool                Flag to print output

        """

        ### Run tests on input parameters
        if len(imls)!=len(apoes): 
            raise ValueError('The hazard curve parameters should match in length!')
        
        self.imls = imls
        self.apoes  = apoes
        self.target_probability   = target_probability
        self.target_frequency   = target_frequency
        self.beta = beta
        self.pflag = pflag 
        
               
    def compute_hazard_derivative(self, imls, apoes):
        """
        Compute the derivative of the hazard curve with respect to spectral acceleration (SA).
        """
        
        hazard_derivative = -np.gradient(apoes, imls)  # Negative slope of the hazard curve
        return hazard_derivative
    
    def find_uhgm(self, imls, apoes, target_frequency):
        """
        Find the spectral acceleration corresponding to the target frequency (UHGM).
        """
        return np.interp(target_frequency, apoes[::-1], imls[::-1])


    def fragility_curve(self, imls, rtgm, beta, pc_collapse=0.1):
        """
        Calculate fragility curve values based on the lognormal distribution.
        """
        phi_inverse = norm.ppf(pc_collapse)
        ln_rtgm_offset = np.log(rtgm) - phi_inverse * beta
        ln_sa = np.log(imls)
        return norm.cdf((ln_sa - ln_rtgm_offset) / beta)

        
    def compute_collapse_risk(self, fragility_values, hazard_derivative_values, delta_imls):
        """
        Compute the collapse risk (lambda_c) by integrating fragility values with hazard curve derivatives.
        """
        risk_density = fragility_values * np.abs(hazard_derivative_values)  # Collapse risk density
        lambda_c = 5000 * np.sum(risk_density * delta_imls)  # Integrate over all SA values
        return lambda_c, risk_density
 
    def compute_rtgm(self,imls, apoes, target_probability, beta, initial_rtgm, tolerance):
        """
        Iteratively compute RTGM until the computed collapse probability matches the target.
        """
        dm = 0.005
        delta_sa = dm  # Fixed increment in SA
        hazard_derivative = self.compute_hazard_derivative(imls, apoes)
        rtgm = initial_rtgm
        iteration = 0
        current_pc = 0
        rtgm_values = []
        fragility_curves = []
        risk_densities = []
        while True:
            # Compute fragility curve
            fragility_values = self.fragility_curve(imls, rtgm, beta=beta)
            # Compute collapse risk
            current_pc, risk_density = self.compute_collapse_risk(fragility_values, hazard_derivative, delta_sa)
            # Save for plotting
            rtgm_values.append(rtgm)
            fragility_curves.append(fragility_values)
            risk_densities.append(risk_density)
            # Print RTGM and PC at each iteration
            print(f"Iteration {iteration + 1}: RTGM = {rtgm:.4f}g, PC = {current_pc:.6f}")
            # Check if current PC is within tolerance
            if abs(current_pc - target_probability) <= tolerance:
                break  # Stop iteration if target is achieved
            # Adjust RTGM
            if current_pc > target_probability:
                rtgm *= 1.05  # Increase RTGM
            else:
                rtgm *= 0.95  # Decrease RTGM
            iteration += 1
            if iteration > 1000:
                print("Max iterations reached. Convergence not achieved.")
                break
        return rtgm, current_pc, iteration, rtgm_values, fragility_curves, risk_densities
        
    
    def run_calculations(self):
        
        # Step 1: Hazard Curve Extrapolation and Log-log interpolation of the hazard curve
        dm = 0.005
        im_extrapolated = np.arange(0.01, 2.0, dm)  # Uniform increments of 0.005
        hazard_interpolator = interp1d(self.imls, np.log(self.apoes), kind="linear", fill_value="extrapolate")
        hazard_extrapolated = np.exp(hazard_interpolator(im_extrapolated))  # Convert back to linear scale
        
        # Step 2: Compute Derivative of Hazard Curve and calculate uniform hazard ground motion
        hazard_derivative = self.compute_hazard_derivative(im_extrapolated, hazard_extrapolated)
        uhgm = self.find_uhgm(self.imls, self.apoes, self.target_frequency)
        
        # Step 3: Define fragility curve, compute collapse risk and run iterative RTGM Adjustment
        # Perform Iterative Calculation
                                                                                               #compute_rtgm(imls,                          apoes,      target_probability,      beta, initial_rtgm, tolerance)
        final_rtgm, final_pc, iterations, rtgm_values, fragility_curves, risk_densities = self.compute_rtgm(im_extrapolated, hazard_extrapolated, self.target_probability, self.beta, uhgm, self.target_probability/10)
        
        # Step 4: Calculate the risk coefficient
        risk_coefficient = final_rtgm/uhgm
                
        if self.pflag:
            
            # Plot Hazard Curve
            plt.figure(figsize=(10, 6))
            plt.loglog(im_extrapolated, hazard_extrapolated, 'k-', label="Extrapolated Hazard Curve")
            plt.scatter(self.imls, self.apoes, color="red", label="Original Data")
            plt.xlabel("Intensity Measure Level (g)", color = 'blue')
            plt.ylabel("Annual Frequency of Exceedance", color = 'blue')
            plt.title("Hazard Curve")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.legend()
            plt.show()
            
            
            # Plot Fragility Curves Across Iterations
            plt.figure(figsize=(10, 6))
            for i, (rtgm, fragility_values) in enumerate(zip(rtgm_values, fragility_curves)):
                plt.semilogx(
                    im_extrapolated,
                    fragility_values,
                    label=f"Iteration {i+1} (RTGM = {rtgm:.2f}g)"
                )
            plt.axhline(0.1, linestyle="--", color="gray", label="10% Conditional Probability of Collapse")
            plt.xlabel("Spectral Response Acceleration (g)")
            plt.ylabel("Conditional Failure Probability")
            plt.title("Fragility Curves Across Iterations")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.legend()
            plt.show()
            # Plot Collapse Risk Density Across Iterations
            plt.figure(figsize=(10, 6))
            for i, (rtgm, risk_density) in enumerate(zip(rtgm_values, risk_densities)):
                plt.semilogx(
                    im_extrapolated,
                    risk_density,
                    label=f"Iteration {i+1} (RTGM = {rtgm:.2f}g)"
                )
            plt.xlabel("Spectral Response Acceleration (g)")
            plt.ylabel("Collapse Risk Density")
            plt.title("Collapse Risk Density Across Iterations")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.legend()
            plt.show()
            # Final Results
            print(f"Final RTGM: {final_rtgm:.4f}g")
            print(f"Final Probability of Collapse (PC): {final_pc:.6f}")
            print(f"Iterations: {iterations}")
            
        return uhgm, final_rtgm, risk_coefficient 

    

#%% EXAMPLE RUN TO TEST CLASS

def csv_to_numpy(csv):
    return pd.read_csv(csv, skiprows=1).to_numpy()    

def get_apoe(poe):
    return -math.log(1-poe)/50

def get_imls_from_csv(csv):
    dataframe = pd.read_csv(csv, skiprows = 1) # load the data in the hazard files
    imls = []    
    headers = list(dataframe.columns.values)
    imls_strings = headers[3:] # this might change depending on the version of hazard output
    for i in range(len(imls_strings)):
        imls.append(float(imls_strings[i].replace('poe-','')))
    imls = np.array(imls)
    return imls 

def process_hazard_files(csv):
        
    # process intensity measure levels, probability of exceedance in 50 years, and annual probabilities
    imls = get_imls_from_csv(csv)    
    haz_array = csv_to_numpy(csv)
    poe = haz_array[:,3:]
    
    apoe = np.zeros((poe.shape[0],poe.shape[1]))
    for i in range(poe.shape[0]):
        for j in range(poe.shape[1]):
            apoe[i,j] = get_apoe(poe[i,j])
    
    # create and store the latitudes and longitudes
    lonlat = np.zeros((haz_array.shape[0],2))
    lonlat[:,0] = haz_array[:,0]
    lonlat[:,1] = haz_array[:,1]

    return lonlat, apoe, poe, imls    


# Process the hazard file for PGA
file = '/Users/mnafeh/Desktop/GEM/Projects/1. METIS/1.Data/Hazard/hazard_curve-mean-PGA_53559.csv'
lonlat, _, apoes, imls = process_hazard_files(file)

# Load the locations: let's consider the NPP locations from the METIS project
NPP_locs = []
NPP = pd.read_csv('/Users/mnafeh/Desktop/GEM/Projects/1. METIS/1.Data/Locations/selected_locations.csv')
name = NPP['name'].tolist()
country = NPP['country_long']
latitude = NPP['latitude'].to_numpy()
longitude = NPP['longitude'].to_numpy()
indices = []

NPP_locs = []
for i in range(len(NPP)):    
    NPP_locs.append((NPP['longitude'][i],NPP['latitude'][i]))


for i in range(len(NPP_locs)):
    
    tree = spatial.KDTree(lonlat)
    index = tree.query(NPP_locs[i])[1]
    indices.append(index)




## Class input parameters
target_frequency     = 1/2475     # Mean annual hazard frequency (corresponds to 1/return period, for nuclear power plant applications, ASCE 43-05 considers a return period of 10000 for SDC-5 NPPs)
target_probability   = 0.05       # Target acceptable annual probability of failure (ASCE 43-05 considers an acceptable failure probability of 1e-5 for SDC-5 NPPs) 
beta = 0.8                        # Dispersion in fragility function
pflag = True                      # Flag to plot outputs (hazard curve, uhs iterations, fragility iterations)


# Loop over the locations
for i in range(len(NPP_locs)):

    current_imls = imls
    current_apoes = apoes[indices[i],:]

    # Initialise the class 
    rt_calculator = risk_targetor(current_imls, current_apoes, target_probability, target_frequency, beta, pflag=True)
    
    # Run the calculations
    uhgm, final_rtgm, risk_coefficient = rt_calculator.run_calculations()













