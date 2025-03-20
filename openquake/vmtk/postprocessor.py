import numpy as np
import pandas as pd
#import statsmodels.api as sm
from scipy import stats, optimize
from scipy.stats import norm, lognorm 
#from statsmodels.miscmodels.ordinal_model import OrderedModel

class postprocessor():

    """
    Class for post-processing results of nonlinear time-history analysis, including fragility and vulnerability analysis.  
    
    This class provides methods to compute fragility functions, perform cloud and multiple stripe analyses, 
    and calculate vulnerability functions and average annual losses. It supports various fragility fitting 
    methods, including lognormal, probit, logit, and ordinal models. The class also includes functionality 
    to handle uncertainty and variability in the analysis.
    
    Methods
    -------
    calculate_lognormal_fragility(theta, sigma_record2record, sigma_build2build=0.30, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3))
        Computes the probability of exceeding a damage state using a lognormal cumulative distribution function (CDF).
    
    calculate_rotated_fragility(theta, percentile, sigma_record2record, sigma_build2build=0.30, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3))
        Calculates a rotated fragility function based on a lognormal CDF, adjusting the median intensity to align with a specified target percentile.
    
    calculate_glm_fragility(imls, edps, damage_thresholds, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3), fragility_method='logit')
        Computes non-parametric fragility functions using Generalized Linear Models (GLM) with either a Logit or Probit link function.
    
    calculate_ordinal_fragility(imls, edps, damage_thresholds, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3))
        Fits an ordinal (cumulative) probit model to estimate fragility curves for different damage states.
    
    do_cloud_analysis(imls, edps, damage_thresholds, lower_limit, censored_limit, sigma_build2build=0.3, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3), fragility_rotation=False, rotation_percentile=0.1, fragility_method='lognormal')
        Perform a censored cloud analysis to assess fragility functions for a set of engineering demand parameters (EDPs) and intensity measure levels (IMLs).
    
    do_multiple_stripe_analysis(imls, edps, damage_thresholds, sigma_build2build=0.3, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3), fragility_rotation=False, rotation_percentile=0.10)
        Perform maximum likelihood estimation (MLE) for fragility curve fitting following a multiple stripe analysis.
    
    calculate_sigma_loss(loss)
        Calculate the uncertainty in the loss estimates based on the method proposed in Silva (2019).
    
    get_vulnerability_function(poes, consequence_model, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3), uncertainty=True)
        Calculate the vulnerability function given the probabilities of exceedance and a consequence model.
    
    calculate_average_annual_damage_probability(fragility_array, hazard_array, return_period=1, max_return_period=5000)
        Calculate the Average Annual Damage State Probability (AADP) based on fragility and hazard curves.
    
    calculate_average_annual_loss(vulnerability_array, hazard_array, return_period=1, max_return_period=5000)
        Calculate the Average Annual Loss (AAL) based on vulnerability and hazard curves.
    
    """  
    
    def __init__(self):
        pass                
            
    def calculate_lognormal_fragility(self,
                                      theta,
                                      sigma_record2record,
                                      sigma_build2build = 0.30,
                                      intensities = np.round(np.geomspace(0.05, 10.0, 50), 3)):
        """
        Computes the probability of exceeding a damage state using a lognormal cumulative distribution function (CDF).
        
        Parameters
        ----------
        theta : float
            The median seismic intensity corresponding to an EDP-based damage threshold.
            
        sigma_record2record : float
            The logarithmic standard deviation representing record-to-record variability.
            
        sigma_build2build : float, optional
            The logarithmic standard deviation representing building-to-building (or model) variability.
            Default value is 0.30.
            
        intensities : array-like, optional
            The set of intensity measure (IM) levels for which exceedance probabilities will be computed.
            Default is a geometric sequence from 0.05 to 10.0 with 50 points.
    
        Returns
        -------
        poes : numpy.ndarray
            An array of exceedance probabilities corresponding to each intensity measure in `intensities`.
        
        References
        -----
        1) Baker JW. Efficient Analytical Fragility Function Fitting Using Dynamic Structural Analysis. 
        Earthquake Spectra. 2015;31(1):579-599. doi:10.1193/021113EQS025M
        
        2) Singhal A, Kiremidjian AS. Method for probabilistic evaluation of seismic structural damage. 
        Journal of Structural Engineering 1996; 122: 1459–1467. DOI:10.1061/(ASCE)0733-9445(1996)122:12(1459)
        
        3) Lallemant, D., Kiremidjian, A., and Burton, H. (2015), Statistical procedures for developing 
        earthquake damage fragility curves. Earthquake Engng Struct. Dyn., 44, 1373–1389. doi: 10.1002/eqe.2522.
        
        4) Bird JF, Bommer JJ, Bray JD, Sancio R, Spence RJS. Comparing loss estimation with observed damage in a zone
        of ground failure: a study of the 1999 Kocaeli Earthquake in Turkey. Bulletin of Earthquake Engineering 2004; 2:
        329–360. DOI: 10.1007/s10518-004-3804-0
        
        """        
        
        # Calculate the total uncertainty
        beta_total = np.sqrt(sigma_record2record**2+sigma_build2build**2)
        
        # Calculate probabilities of exceedance for a range of intensity measure levels
        return lognorm.cdf(intensities, s=beta_total, loc=0, scale=theta) 

    def calculate_rotated_fragility(self,
                                    theta, 
                                    percentile,
                                    sigma_record2record, 
                                    sigma_build2build = 0.30,
                                    intensities = np.round(np.geomspace(0.05, 10.0, 50), 3)):
        """
        Calculates a rotated fragility function based on a lognormal cumulative distribution function (CDF),
        adjusting the median intensity to align with a specified target percentile.
    
        This function modifies the median intensity based on the desired target percentile and total uncertainty 
        (considering both record-to-record variability and modeling variability). The resulting rotated fragility 
        curve represents the damage exceedance probabilities for a range of intensity measure levels, as defined 
        by the lognormal distribution.
        
        ----------
        Parameters
        ----------
        theta : float
            The median seismic intensity corresponding to the edp-based damage threshold.
    
        percentile : float
            The target percentile for fragility function rotation. This value corresponds to the desired 
            percentile (e.g., 0.2 corresponds to the 20th percentile of the fragility curve). The curve is adjusted 
            such that this percentile aligns with the calculated fragility function.
    
        sigma_record2record : float
            The uncertainty associated with record-to-record variability in the seismic records used to derive the fragility.
    
        sigma_build2build : float, optional, default=0.30
            The uncertainty associated with modeling variability between different buildings or building types.
    
        intensities : array-like, optional, default=np.round(np.geomspace(0.05, 10.0, 50), 3)
            A list or array of intensity measure levels at which to evaluate the fragility function, typically representing
            seismic intensity levels (e.g., spectral acceleration). The default is a geometric space ranging from 0.05 to 10.0.
    
        -------
        Returns
        -------
        theta_prime : float
            The new median intensity after the rotation based on the specified percentile.
    
        beta_total : float
            The total standard deviation of the lognormal distribution, calculated from both record-to-record and 
            building-to-building (modelling) uncertainties.
    
        poes : array-like
            The probabilities of exceedance (fragility values) corresponding to the input intensity measure levels.
            This is the lognormal CDF evaluated at the given intensities with the rotated median and combined uncertainty.

        ----------
        References
        ----------
        1) Porter, K. (2017), "When Addressing Epistemic Uncertainty in a Lognormal Fragility Function, 
        How Should One Adjust the Median?", Proceedings of the 16th World Conference on Earthquake Engineering 
        (16WCEE), Santiago, Chile.

        """
                
        # Calculate the combined logarithmic standard deviation (total uncertainty)
        beta_total = np.sqrt(sigma_record2record**2 + sigma_build2build**2)
        
        # Adjust the median intensity based on the target percentile
        theta_prime = theta * np.exp(-stats.norm.ppf(percentile) * (beta_total - sigma_record2record))
        
        # Calculate and return the rotated lognormal CDF (probabilities of exceedance) for the given intensities
        return theta_prime, beta_total, stats.lognorm(s=beta_total, scale=theta_prime).cdf(intensities)


    def calculate_glm_fragility(self,
                                imls,
                                edps,
                                damage_thresholds,
                                intensities=np.round(np.geomspace(0.05, 10.0, 50), 3),
                                fragility_method = 'logit'):

        """
        Computes non-parametric fragility functions using Generalized Linear Models (GLM) with 
        either a Logit or Probit link function.
    
        Parameters:
        -----------
        imls : array-like
            Intensity Measure Levels (IMLs) corresponding to each observation.
            
        edps : array-like
            Engineering Demand Parameters (EDPs) representing structural response values.
            
        damage_thresholds : array-like
            List of thresholds defining different damage states.
            
        intensities : array-like, optional
            Intensity measure values at which probabilities of exceedance (PoEs) are evaluated.
            Defaults to np.round(np.geomspace(0.05, 10.0, 50), 3).
            
        fragility_method : str, optional
            Specifies the GLM model to be used for fragility function fitting.
            Options:
            - 'logit' (default): Uses a logistic regression model.
            - 'probit': Uses a probit regression model.
    
        Returns:
        --------
        poes : ndarray
            A 2D array where each column represents the probability of exceeding a 
            specific damage state at each intensity level.
    
        References:
        ------
        1) Charvet, I., Ioannou, I., Rossetto, T., Suppasri, A., and Imamura, F.: Empirical fragility 
        assessment of buildings affected by the 2011 Great East Japan tsunami using improved statistical models, 
        Nat. Hazards, 73, 951–973, 2014. 
        
        2) Lahcene, E., Ioannou, I., Suppasri, A., Pakoksung, K., Paulik, R., Syamsidik, S., Bouchette, F., 
        and Imamura, F.: Characteristics of building fragility curves for seismic and non-seismic tsunamis: 
        case studies of the 2018 Sunda Strait, 2018 Sulawesi–Palu, and 2004 Indian Ocean tsunamis, 
        Nat. Hazards Earth Syst. Sci., 21, 2313–2344, https://doi.org/10.5194/nhess-21-2313-2021, 2021.
        
        3) Lallemant, D., Kiremidjian, A., and Burton, H. (2015), Statistical procedures for developing 
        earthquake damage fragility curves. Earthquake Engng Struct. Dyn., 44, 1373–1389. doi: 10.1002/eqe.2522.
        
        4) Jalayer, F., Ebrahamian, H., Trevlopoulos, K., and Bradley, B. (2023). Empirical tsunami fragility modelling 
        for hierarchical damage levels. Natural Hazards and Earth System Sciences, 23(2), 909–931. 
        https://doi.org/10.5194/nhess-23-909-2023
        
        """
        
        # Create probabilities of exceedance array
        poes = np.zeros((len(intensities),len(damage_thresholds)))
        
        for ds, current_threshold in enumerate(damage_thresholds):
            
            # Count exceedances
            exceedances = [1 if edp>damage_thresholds[ds] else 0 for edp in edps]
            
            # Assemble dictionary containing log of IMs and binary damage state assignments 
            data = {'IM': np.log(imls),
                    'Damage': exceedances} 
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Add a constant for the intercept term
            X = sm.add_constant(df['IM'])
            y = df['Damage']
            
            if fragility_method.lower() == 'probit':
    
                # Fit the Probit GLM model 
                probit_model = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.Probit()))
                probit_results = probit_model.fit()
                
                # Generate a range of IM values for plotting
                log_IM_range = np.log(intensities)
                X_range = sm.add_constant(log_IM_range)
                
                # Predict probabilities using the Probit GLM model
                poes[:,ds] = probit_results.predict(X_range)
            
            elif fragility_method.lower() == 'logit':
            
                # Fit the Logit GLM model 
                logit_model = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.Logit()))
                logit_results = logit_model.fit()
                
                # Generate a range of IM values for plotting
                log_IM_range = np.log(intensities)
                X_range = sm.add_constant(log_IM_range)
                
                # Predict probabilities using the Probit GLM model
                poes[:,ds] = logit_results.predict(X_range)
            
        return poes

    def calculate_ordinal_fragility(self,
                                    imls, 
                                    edps, 
                                    damage_thresholds,
                                    intensities=np.round(np.geomspace(0.05, 10.0, 50), 3)):
        """
        Fits an ordinal (cumulative) probit model to estimate fragility curves for different damage states.
    
        This function estimates the probability of exceeding various damage states using an ordinal 
        regression model based on observed Engineering Demand Parameters (EDPs) and corresponding 
        Intensity Measure Levels (IMLs).
    
        Parameters
        ----------
        imls : array-like
            Intensity measure levels corresponding to the observed EDPs.
            
        edps : array-like
            Engineering Demand Parameters (EDPs) representing structural responses.
            
        damage_thresholds : array-like
            Damage state thresholds for classifying exceedance levels.
            
        intensities : array-like, optional
            Intensity measure levels for which fragility curves are evaluated (default: np.geomspace(0.05, 10.0, 50)).
    
        Returns
        -------
        poes : numpy.ndarray
            A 2D array of exceedance probabilities (CDF values) for each intensity level.
            Shape: (len(intensities), len(damage_thresholds) + 1), where the last column 
            represents the probability of exceeding the highest damage state.
    
        References
        -----
        1) Lallemant, D., Kiremidjian, A., and Burton, H. (2015), Statistical procedures for developing 
        earthquake damage fragility curves. Earthquake Engng Struct. Dyn., 44, 1373–1389. doi: 10.1002/eqe.2522.
        
        2) Nguyen, M. and Lallemant, D. (2022), Order Matters: The Benefits of Ordinal Fragility Curves for Damage and Loss Estimation. Risk Analysis, 42: 1136-1148. https://doi.org/10.1111/risa.13815

        """
    
        # Create probabilities of exceedance array
        poes = np.zeros((len(intensities), len(damage_thresholds) + 1))  # +1 to include the highest damage state
        
        # Initialize damage state assignments
        damage_states = np.zeros(len(edps), dtype=int)
        
        # Loop over each EDP and determine the highest exceeded damage state
        for i, edp in enumerate(edps):
            exceeded = np.where(edp > damage_thresholds)[0]  # Indices where EDP exceeds thresholds
            damage_states[i] = exceeded[-1] + 1 if exceeded.size > 0 else 0  # Assign highest exceeded state (0-based)
              
        # Assemble DataFrame containing log(IM) and damage state assignment
        df = pd.DataFrame({'IM': np.log(imls), 'Damage State': damage_states})
    
        # Fit the Cumulative Probit Model 
        X_ordinal = df[['IM']]
        y_ordinal = df['Damage State']
    
        # Create and fit the OrderedModel
        ordinal_model = OrderedModel(y_ordinal, X_ordinal, distr='probit')
        ordinal_results = ordinal_model.fit(method='bfgs', disp=False)  # Silent optimization
        
        # Generate log-transformed IM values for prediction
        log_IM_range = np.log(intensities)
        X_range_ordinal = pd.DataFrame({'IM': log_IM_range})
    
        # Predict probabilities for each damage state (PMF)
        pmf_values = ordinal_results.predict(X_range_ordinal)  # Shape: (len(intensities), num_damage_states)
    
        # Convert PMF to CDF (probabilities of exceedance) by cumulative sum across damage states
        poes = 1 - np.cumsum(pmf_values, axis=1)  # Cumulative sum along damage state axis
        
        return poes.values

    def do_cloud_analysis(self,
                          imls, 
                          edps, 
                          damage_thresholds, 
                          lower_limit, 
                          censored_limit, 
                          sigma_build2build=0.3, 
                          intensities=np.round(np.geomspace(0.05, 10.0, 50), 3),
                          fragility_rotation=False,
                          rotation_percentile=0.1,
                          fragility_method='lognormal'):
        """
        Perform a censored cloud analysis to assess fragility functions for a set of engineering demand parameters (EDPs) 
        and intensity measure levels (IMLs). This function processes the cloud analysis and fits regression models, 
        considering both lower and upper limits for censored data. The method is used for deriving fragility functions 
        such as those in ESRM20.
    
        This function allows the application of various fragility function fitting methods (parametric and non-parametric) 
        and optionally rotates the fragility function around a given percentile.
    
        Parameters:
        -----------
        imls : list or array
            A list or array of intensity measure levels (IMLs), representing the levels of seismic intensity considered 
            for the analysis.
            
        edps : list or array
            A list or array of engineering demand parameters (EDPs) such as maximum interstory drifts, maximum peak 
            floor acceleration, or top displacements that are used to assess the structural response to the seismic event.
            
        damage_thresholds : list
            A list of damage thresholds associated with different levels of damage (e.g., slight, moderate, extensive, and 
            complete). These thresholds are used to classify the damage states based on the corresponding EDP values.
    
        lower_limit : float
            The minimum value of EDP below which cloud records are excluded. This is typically set to a small value, 
            such as 0.1 times the yield capacity, to avoid records with no damage.
            
        censored_limit : float
            The maximum value of EDP above which cloud records are excluded. This is typically set to a value like 
            1.5 times the ultimate capacity to filter out records corresponding to collapse.
            
        sigma_build2build : float, optional, default=0.3
            The building-to-building variability or modeling uncertainty. This represents the variation in response 
            from one building to another, and is used to model uncertainty in the fragility function.
            
        intensities : array, optional, default=np.geomspace(0.05, 10.0, 50)
            An array of intensity measure levels used to sample and evaluate the fragility functions. By default, 
            this is set to a logarithmic space between 0.05 and 10.0 with 50 points.
    
        fragility_rotation : bool, optional, default=False
            A boolean flag to indicate whether or not the fragility function should be rotated about a target percentile.
            If set to `True`, the function will rotate the fragility curve to match the given percentile.
            
        rotation_percentile : float, optional, default=0.1
            The target percentile (between 0 and 1) around which the fragility function will be rotated, if 
            `fragility_rotation` is `True`. For example, a value of 0.1 means a rotation to the 10th percentile.
            
        fragility_method : str, optional, default='probit'
            The method used to fit the fragility function. Options include:
            - 'probit': Probit regression model (default).
            - 'logit': Logit regression model.
            - 'ordinal': Ordinal regression model.
            - 'lognormal': Lognormal distribution model for the fragility function.
            
        Returns:
        --------
        cloud_dict : dict
            A dictionary containing the outputs of the cloud analysis, which includes:
            - 'cloud inputs': Input data used in the analysis (IMLs, EDPs, thresholds, limits).
            - 'fragility functions': Results of the fragility function fitting, including predicted exceedance probabilities (poes), 
              fragility method used, and related parameters.
            - 'regression coefficients': Fitted regression coefficients (b1, b0), sigma values, and fitted data for the model.
            """
        
        # Convert inputs to numpy arrays
        imls = np.asarray(imls)
        edps = np.asarray(edps)
        
        # Compute exceedance probabilities using the specified fragility method
        if fragility_method in ['probit', 'logit']:
            
            # Get the probabilities of exceedance
            poes = self.calculate_glm_fragility(imls, edps, damage_thresholds, fragility_method=fragility_method)
            
            # Create the dictionary
            cloud_dict = {
                
                # Add a nested dictionary for the inputs of the regression
                'cloud inputs': {
                    'imls': imls,                            # Store the intensity measure levels (cloud)
                    'edps': edps,                            # Store the engineering demand parameters (cloud)
                    'lower_limit': None,                     # Store the lower limit for censored regression
                    'upper_limit': None,                     # Store the upper limit for censored regression
                    'damage_thresholds': damage_thresholds   # Store the demand-based damage state thresholds
                    },
                
                # Add a nested dictionary for fragility functions parameters
                'fragility': {
                    'fragility_method': fragility_method.lower(), # Store the fragility fitting methodology
                    'intensities': intensities,                   # Store the intensities used for sampling fragility functions
                    'poes': poes,                                 # Store the probabilities of damage state exceedance
                    'medians': None,                              # Store the median seismic intensities
                    'sigma_record2record': None,                  # Store the record-to-record variability
                    'sigma_build2build': None,                    # Store the modelling uncertainty
                    'betas_total': None                           # Store the total variability accounting for record-to-record and modelling uncertainties 
                    },
                
                # Add a nested dictionary for regression coefficients
                'regression': {
                    'b1': None,         # Store 'b1' coefficient
                    'b0': None,         # Store 'b0' coefficient
                    'sigma': None,      # Store 'sigma' value
                    'fitted_x': None,   # Store the fitted x-values
                    'fitted_y': None    # Store the fitted y-values
                }
            }

            
        elif fragility_method.lower() == 'ordinal':

            # Compute exceedance probabilities using the specified fragility method            
            poes = self.fit_ordinal_fragility(imls, edps, damage_thresholds)
            
            # Create the dictionary
            cloud_dict = {
                
                # Add a nested dictionary for the inputs of the regression
                'cloud inputs': {
                    'imls': imls,                            # Store the intensity measure levels (cloud)
                    'edps': edps,                            # Store the engineering demand parameters (cloud)
                    'lower_limit': None,                     # Store the lower limit for censored regression
                    'upper_limit': None,                     # Store the upper limit for censored regression
                    'damage_thresholds': damage_thresholds   # Store the demand-based damage state thresholds
                    },
                
                # Add a nested dictionary for fragility functions parameters
                'fragility': {
                    'fragility_method': fragility_method.lower(), # Store the fragility fitting methodology
                    'intensities': intensities,                   # Store the intensities used for sampling fragility functions
                    'poes': poes,                                 # Store the probabilities of damage state exceedance
                    'medians': None,                              # Store the median seismic intensities
                    'sigma_record2record': None,                  # Store the record-to-record variability
                    'sigma_build2build': None,                    # Store the modelling uncertainty
                    'betas_total': None                           # Store the total variability accounting for record-to-record and modelling uncertainties 
                    },
                
                # Add a nested dictionary for regression coefficients
                'regression': {
                    'b1': None,         # Store 'b1' coefficient
                    'b0': None,         # Store 'b0' coefficient
                    'sigma': None,      # Store 'sigma' value
                    'fitted_x': None,   # Store the fitted x-values
                    'fitted_y': None    # Store the fitted y-values
                }
            }
            
        elif fragility_method.lower() == 'lognormal':
            
            # define the arrays for the regression
            x_array=np.log(imls)
            y_array=edps
            
            # remove displacements below lower limit
            bool_y_lowdisp=edps>=lower_limit
            x_array = x_array[bool_y_lowdisp]
            y_array = y_array[bool_y_lowdisp]
    
            # checks if the y value is above the censored limit
            bool_is_censored=y_array>=censored_limit
            bool_is_not_censored=y_array<censored_limit
            
            # creates an array where all the censored values are set to the limit
            observed=np.log((y_array*bool_is_not_censored)+(censored_limit*bool_is_censored))
            
            y_array=np.log(edps)
            
            def func(x):
                  p = np.array([norm.pdf(observed[i], loc=x[1]+x[0]*x_array[i], scale=x[2]) for i in range(len(observed))],dtype=float)
                  return -np.sum(np.log(p))
            sol1=optimize.fmin(func,[1,1,1],disp=False)
            
            def func2(x):
                  p1 = np.array([norm.pdf(observed[i], loc=x[1]+x[0]*x_array[i], scale=x[2]) for i in range(len(observed)) if bool_is_censored[i]==0],dtype=float)
                  p2 = np.array([1-norm.cdf(observed[i], loc=x[1]+x[0]*x_array[i], scale=x[2]) for i in range(len(observed)) if bool_is_censored[i]==1],dtype=float)
                  return -np.sum(np.log(p1[p1 != 0]))-np.sum(np.log(p2[p2 != 0]))
            
            p_cens=optimize.fmin(func2,[sol1[0],sol1[1],sol1[2]],disp=False)
    
            # Regression fit
            xvec = np.linspace(np.log(min(imls)), np.log(max(imls)), 100)
            yvec = p_cens[0] * xvec + p_cens[1]
            
            # Compute fragility parameters from the regressed fit
            thetas               = np.exp((np.log(damage_thresholds) - p_cens[1]) / p_cens[0])                                       # Median intensities
            sigmas_record2record = np.full(len(damage_thresholds), p_cens[2] / p_cens[0])                                            # Record-to-record variability
            sigmas_build2build   = np.full(len(damage_thresholds), sigma_build2build)                                                # Modelling uncertainty
            betas_total          = np.full(len(damage_thresholds), np.sqrt((p_cens[2] / p_cens[0])**2 + sigma_build2build**2))       # Total dispersion
    
            # Compute probabilities of exceedance
            if fragility_rotation:
                
                fragility_method == f'lognormal - rotated around the {rotation_percentile}th percentile'
                poes = np.array([self.calculate_rotated_fragility(theta,
                                                                  rotation_percentile, 
                                                                  sigma_record2record, 
                                                                  sigma_build2build = sigma_build2build) for theta, sigma_record2record in zip(thetas, sigmas_record2record)]).T        
            else:            
                poes = np.array([self.calculate_lognormal_fragility(theta, 
                                                                    sigma_record2record, 
                                                                    sigma_build2build = sigma_build2build) for theta, sigma_record2record in zip(thetas, sigmas_record2record)]).T

            # Create the dictionary
            cloud_dict = {
                
                # Add a nested dictionary for the inputs of the regression
                'cloud inputs': {
                    'imls': imls,                            # Store the intensity measure levels (cloud)
                    'edps': edps,                            # Store the engineering demand parameters (cloud)
                    'lower_limit': None,                     # Store the lower limit for censored regression
                    'upper_limit': None,                     # Store the upper limit for censored regression
                    'damage_thresholds': damage_thresholds   # Store the demand-based damage state thresholds
                    },
                
                # Add a nested dictionary for fragility functions parameters
                'fragility': {
                    'fragility_method': fragility_method.lower(), # Store the fragility fitting methodology
                    'intensities': intensities,                   # Store the intensities used for sampling fragility functions
                    'poes': poes,                                 # Store the probabilities of damage state exceedance
                    'medians': thetas,                            # Store the median seismic intensities
                    'sigma_record2record': sigmas_record2record,  # Store the record-to-record variability
                    'sigma_build2build': sigmas_build2build,      # Store the modelling uncertainty
                    'betas_total': betas_total                    # Store the total variability accounting for record-to-record and modelling uncertainties 
                    },
                
                # Add a nested dictionary for regression coefficients
                'regression': {
                    'b1': p_cens[0],            # Store 'b1' coefficient
                    'b0': p_cens[1],            # Store 'b0' coefficient
                    'sigma': p_cens[2],         # Store 'sigma' value
                    'fitted_x': np.exp(xvec),   # Store the fitted x-values
                    'fitted_y': np.exp(yvec)    # Store the fitted y-values
                }
            }
        
        return cloud_dict
    
    def do_multiple_stripe_analysis(self,
                                     imls, 
                                     edps, 
                                     damage_thresholds, 
                                     sigma_build2build=0.3, 
                                     intensities=np.round(np.geomspace(0.05, 10.0, 50), 3),
                                     fragility_rotation=False,
                                     rotation_percentile=0.10):
        """
        Perform maximum likelihood estimation (MLE) for fragility curve fitting following a multiple stripe analysis.
        This method calculates the fragility function by fitting to the provided intensity measure levels (IMLs) 
        and engineering demand parameters (EDPs) "stripes", with the option to rotate the fragility curve around 
        a target percentile.
    
        The method is useful for deriving fragility functions by determining the probability 
        of exceedance for various damage states based on the provided data.
    
        Parameters:
        -----------
        imls : list or array
            A list or array of intensity measure levels (IMLs) representing the seismic intensity levels used for 
            sampling the fragility functions.
            
        edps : list or array
            A list or array of engineering demand parameters (EDPs), which describe the structural response to 
            seismic events. Examples include maximum interstorey drifts, maximum peak floor acceleration, or top 
            displacements.
    
        damage_thresholds : list
            A list of EDP-based damage thresholds that correspond to different levels of structural damage, such 
            as slight, moderate, extensive, and complete. These thresholds help categorize the severity of damage 
            based on EDP values.
    
        sigma_build2build : float, optional, default=0.3
            The building-to-building variability or modeling uncertainty. It accounts for differences in performance 
            between buildings with similar characteristics due to random variations or model uncertainties.
    
        intensities : array, optional, default=np.geomspace(0.05, 10.0, 50)
            An array of intensity measure levels over which the fragility function will be sampled. By default, 
            this is a logarithmic space ranging from 0.05 to 10.0, with 50 sample points.
    
        fragility_rotation : bool, optional, default=False
            A boolean flag that determines whether or not to rotate the fragility curve about a given percentile. 
            If `True`, the fragility curve will be adjusted based on the specified `rotation_percentile`.
    
        rotation_percentile : float, optional, default=0.10
            The target percentile (between 0 and 1) around which the fragility function will be rotated. A value of 
            0.10 corresponds to rotating the curve to the 10th percentile.
    
        Returns:
        --------
        msa_dict : dict
            A dictionary containing the results of the multiple stripe analysis, including:
            - 'medians': The estimated medians of the fragility function.
            - 'dispersions': The estimated dispersions (variability) associated with the fragility function.
            - 'poes': The probabilities of exceedance (damage probabilities) for different damage states.
    
        Notes:
        ------
        This method fits the fragility curve using MLE, which minimizes the difference between observed and predicted 
        exceedance probabilities. The option for fragility curve rotation allows for adjusting the curve to better 
        match the expected percentile of damage occurrence, offering greater flexibility in representing the fragility 
        of the structure.
        """        
        
        def likelihood(a0, imls, num_gmr, num_exc, sigma_build2build):
            """Calculate the negative log-likelihood for MLE optimization"""
            x = imls
            n = num_gmr
            z = num_exc
            np.insert(x, 0, 0.)
            np.insert(n, 0, n[0])
            np.insert(z, 0, 0.)
            eta = a0[0]  # Median (θ)
            beta = a0[1]  # Dispersion (β)
            # Total beta considering build-to-build variability
            beta_total = np.sqrt(beta**2 + sigma_build2build**2)
            p = stats.norm.cdf(np.log(x / eta) / beta_total)
            f = [stats.binom.pmf(z[j], n[j], p[j]) for j in range(len(p))]
            return 1. / sum(np.log(f))
    
        results = {}
        
        # Loop over all damage thresholds
        for threshold in damage_thresholds:
            # Count exceedances for each IM level
            num_exc = np.array([np.sum(edp >= threshold) for edp in edps])
            num_gmr = np.full(len(imls), len(edps[0]))  # Number of ground motions at each IM level
            
            # Perform MLE to fit the fragility curve
            initial_guess = [np.median(imls), 0.5]
            
            # Calculate bounds dynamically based on the IML range
            eta_lower_bound = 0.001 * np.min(imls)
            eta_upper_bound = np.max(imls) * 100
            beta_lower_bound = 0.05
            beta_upper_bound = 1.5
            
            # Ensure that the lower bound is always smaller than the upper bound
            bounds = optimize.Bounds([eta_lower_bound, beta_lower_bound], [eta_upper_bound, beta_upper_bound])
            
            # Minimize negative log-likelihood function
            sol = optimize.minimize(likelihood, initial_guess, args=(imls, num_gmr, num_exc, sigma_build2build), bounds=bounds)
            
            theta = sol.x[0]  # Median (θ)
            sigma_record2record = sol.x[1] # Dispersion (β) due to record-to-record variability
            
            # Store results for each damage threshold
            results[threshold] = (theta, sigma_record2record)
        
        # Calculate probabilities of exceedance for given intensity levels
        poes = np.zeros((len(intensities), len(damage_thresholds)))
        for i, threshold in enumerate(damage_thresholds):
            theta = results[threshold][0]
            sigma_record2record = results[threshold][1]
            
            if fragility_rotation:
                poes[:, i] = self.calculate_rotated_fragility(theta, 
                                                             rotation_percentile, 
                                                             sigma_record2record, 
                                                             sigma_build2build = sigma_build2build)
            else:
                poes[:, i] = self.calculate_lognormal_fragility(theta, 
                                                                   sigma_record2record, 
                                                                   sigma_build2build = sigma_build2build)
                
        
        # Create the msa_dict with all relevant information
        msa_dict = {'imls': imls,                                                                                                  # Input intensity measure levels
                    'edps': edps,                                                                                                  # Input engineering demand parameters
                    'damage_thresholds': damage_thresholds,                                                                        # Input damage thresholds
                    'medians': [results[threshold][0] for threshold in damage_thresholds],                                         # Median seismic intensities (in g)
                    'betas_total': [np.sqrt(results[threshold][1]**2 + sigma_build2build**2) for threshold in damage_thresholds],  # Associated total dispersion (accounting for building-to-building and modelling uncertainties)
                    'poes': poes,                                                                                                  # Probabilities of exceedance of each damage state (DS1 to DSi)
                    'intensities': intensities}                                                                                    # Sampled intensities for fragility analysis 
        
    
        return msa_dict

    def calculate_sigma_loss(self, 
                             loss):
        """
        Calculate the uncertainty in the loss estimates based on the method proposed in Silva (2019), 
        which incorporates the sigma (standard deviation) for loss ratios within seismic vulnerability functions.
    
        This method computes the sigma loss ratio for expected loss ratios and also estimates the parameters 
        of a beta distribution (coefficients a and b), which describe the uncertainty and variability in 
        the loss estimates. The formula used is derived from seismic vulnerability research.
        
        Parameters:
        -----------
        loss : list or array
            A list or array of expected loss ratios. The expected loss ratio represents the proportion of 
            the building's value that is expected to be lost due to an earthquake event, ranging from 0 to 1.
    
        Returns:
        --------
        sigma_loss_ratio : list or array
            The calculated uncertainty (sigma) associated with the mean loss ratio for each input loss value.
            The sigma loss ratio represents the variability of the loss estimates and is computed based on the 
            loss ratios provided.
    
        a_beta_dist : list or array
            The coefficient 'a' of the beta distribution for each loss ratio. This parameter represents the shape 
            of the beta distribution and is used to model the uncertainty in the loss estimates.
    
        b_beta_dist : list or array
            The coefficient 'b' of the beta distribution for each loss ratio. This parameter also represents the 
            shape of the beta distribution, complementing the coefficient 'a' to fully describe the distribution's 
            behavior.
        
        References:
        ----------
        1) Silva, V. (2019) "Uncertainty and correlation in seismic vulnerability functions of building classes." 
        Earthquake Spectra. DOI: 10.1193/013018eqs031m.

        """
        sigma_loss_ratio = np.where(loss == 0, 0,
                                    np.where(loss == 1, 1,
                                             np.sqrt(loss * (-0.7 - 2 * loss + np.sqrt(6.8 * loss + 0.5)))))
        a_beta_dist = np.zeros(loss.shape)
        b_beta_dist = np.zeros(loss.shape)
        
        return sigma_loss_ratio, a_beta_dist, b_beta_dist
            
    def get_vulnerability_function(self,
                                   poes,
                                   consequence_model,
                                   intensities=np.round(np.geomspace(0.05, 10.0, 50), 3),
                                   uncertainty=True):
        """
        Calculate the vulnerability function given the probabilities of exceedance and a consequence model, 
        and optionally compute the uncertainty (coefficient of variation) in the expected loss.
    
        This function computes the expected loss ratios for a range of intensity measure levels (IMLs) 
        based on the probabilities of exceedance and the corresponding consequence model. Additionally, 
        it calculates the coefficient of variation (COV) of the loss ratio if the uncertainty flag is set to True.
        
        Parameters:
        -----------
        poes : array
            An array of probabilities of exceedance associated with the damage states considered. 
            The shape is (number of intensities, number of damage states).
        
        consequence_model : list
            A list of damage-to-loss ratios corresponding to each damage state. It has a length equal 
            to the number of damage states.
        
        intensities : array, optional
            An array of intensity measure levels. The default is a geometric sequence ranging from 
            0.05 to 10.0 with 50 points.
        
        uncertainty : bool, optional
            A flag to indicate whether to calculate (or not) the coefficient of variation associated 
            with Loss|IM. The default is True.
    
        Returns:
        --------
        df : pandas DataFrame
            A DataFrame containing the intensity measure levels (IML), expected loss ratios, and 
            optionally, the coefficient of variation (COV) for each IML. The COV is calculated only 
            if the uncertainty flag is True.
    
        """
        
        # Consistency checks
        if len(consequence_model) != np.size(poes, 1):
            raise Exception('Mismatch between the fragility consequence models!')
        if len(intensities) != np.size(poes, 0):
            raise Exception('Mismatch between the number of IMLs and fragility models!')
        
        # Initialize loss array
        loss = np.zeros([len(intensities),])
        
        # Calculate expected loss ratios
        for i in range(len(intensities)):
            for j in range(0, np.size(poes, 1)):
                if j == (np.size(poes, 1) - 1):
                    loss[i,] = loss[i,] + poes[i, j] * consequence_model[j]
                else:
                    loss[i,] = loss[i,] + (poes[i, j] - poes[i, j + 1]) * consequence_model[j]
        
        # If uncertainty is true, calculate the coefficient of variation
        if uncertainty:
            cov = np.zeros(loss.shape)  
            
            for m in range(loss.shape[0]):                        
                mean_loss_ratio = loss[m]
                
                if mean_loss_ratio < 1e-4:
                    loss[m] = 1e-8
                    cov[m] = 1e-8
                elif np.abs(1 - mean_loss_ratio) < 1e-4:
                    loss[m] = 0.99999
                    cov[m] = 1e-8
                else:
                    # Use the calculate_sigma_loss function for loss-related uncertainty
                    sigma_loss_ratio, a_beta_dist, b_beta_dist = self.calculate_sigma_loss(loss[m])
                    
                    # Coefficient of variation
                    cov[m] = np.min([sigma_loss_ratio / mean_loss_ratio, 0.90 * np.sqrt(mean_loss_ratio * (1 - mean_loss_ratio)) / mean_loss_ratio])
            
            # Store to DataFrame with COV
            df = pd.DataFrame({'IML': intensities,
                               'Loss': loss,
                               'COV': cov})
        
        else:
            # Store to DataFrame without COV
            df = pd.DataFrame({'IML': intensities,
                               'Loss': loss})
        
        return df
            
    def calculate_average_annual_damage_probability(self, 
                                                    fragility_array, 
                                                    hazard_array, 
                                                    return_period=1, 
                                                    max_return_period=5000):
        """
        Calculate the Average Annual Damage State Probability (AADP) based on fragility and hazard curves.
        
        This function estimates the average annual probability of damage states occurring over a given return period,
        using the fragility curve (which relates intensity measure levels to damage state probabilities) and the hazard 
        curve (which relates intensity measure levels to annual rates of exceedance).
    
        The calculation integrates the product of the fragility function and the hazard curve over the specified range
        of intensity measure levels, accounting for the return period and a maximum return period threshold.
        
        Parameters:
        -----------
        fragility_array : 2D array
            A 2D array where the first column contains intensity measure levels, and the second column contains the 
            corresponding probabilities of exceedance for each intensity level.
            
        hazard_array : 2D array
            A 2D array where the first column contains intensity measure levels, and the second column contains the 
            annual rates of exceedance (i.e., the probability of exceedance per year) for each intensity level.
            
        return_period : float, optional, default=1
            The return period used to scale the hazard rate. This is the time span (in years) over which the 
            average annual damage probability is calculated. A typical value is 1 year, but longer periods can be used 
            for multi-year assessments.
            
        max_return_period : float, optional, default=5000
            The maximum return period threshold used to filter out very low hazard rates. The hazard curve is truncated 
            to include only intensity levels with exceedance rates above this threshold.
    
        Returns:
        --------
        average_annual_damage_probability : float
            The average annual damage state probability, calculated by integrating the product of the fragility 
            function and the hazard curve over the given intensity measure levels.
    
        """
        
        # Filter hazard array based on the maximum return period
        max_integration = return_period / max_return_period
        hazard_array = hazard_array[hazard_array[:, 1] >= max_integration]
        
        # Compute mean intensity levels and rate of occurrences
        mean_imls = (hazard_array[:-1, 0] + hazard_array[1:, 0]) / 2
        rate_occ = np.diff(hazard_array[:, 1]) / -return_period
        
        # Define fragility curve for interpolation
        curve_imls = np.hstack(([0], fragility_array[:, 0], [20]))
        curve_ordinates = np.hstack(([0], fragility_array[:, 1], [1]))
        
        # Interpolate fragility curve values at the mean intensity levels
        interpolated_values = np.interp(mean_imls, curve_imls, curve_ordinates)
        
        # Compute the average annual damage probability
        return np.dot(interpolated_values, rate_occ)
        
    def calculate_average_annual_loss(self, 
                                      vulnerability_array, 
                                      hazard_array, 
                                      return_period=1, 
                                      max_return_period=5000):
        """
        Calculate the Average Annual Loss (AAL) based on vulnerability and hazard curves.
        
        This function computes the average annual loss by integrating the product of the vulnerability function 
        (which relates intensity measure levels to loss ratios) and the hazard curve (which relates intensity measure 
        levels to annual rates of exceedance). The result represents the expected average loss over a given return period.
        
        Parameters:
        -----------
        vulnerability_array : 2D array
            A 2D array where the first column contains intensity measure levels, and the second column contains the 
            corresponding loss ratios (representing expected loss relative to the building value or some other metric).
            
        hazard_array : 2D array
            A 2D array where the first column contains intensity measure levels, and the second column contains the 
            annual rates of exceedance (i.e., the probability of exceedance per year) for each intensity level.
            
        return_period : float, optional, default=1
            The return period used to scale the hazard rate. This is the time span (in years) over which the 
            average annual loss is calculated. Typically, this value is 1 year, but longer periods can be used 
            for multi-year assessments.
            
        max_return_period : float, optional, default=5000
            The maximum return period threshold used to filter out very low hazard rates. The hazard curve is truncated 
            to include only intensity levels with exceedance rates above this threshold.
    
        Returns:
        --------
        average_annual_loss : float
            The average annual loss, calculated by integrating the product of the vulnerability function and the 
            hazard curve over the given intensity measure levels. This value represents the expected loss per year.
    
        """
        
        # Filter hazard array based on the maximum return period
        max_integration = return_period / max_return_period
        hazard_array = hazard_array[hazard_array[:, 1] >= max_integration]
        
        # Compute mean intensity levels and rate of occurrences
        mean_imls = (hazard_array[:-1, 0] + hazard_array[1:, 0]) / 2
        rate_occ = np.diff(hazard_array[:, 1]) / -return_period
        
        # Define vulnerability curve for interpolation
        curve_imls = np.hstack(([0], vulnerability_array[:, 0], [20]))
        curve_ordinates = np.hstack(([0], vulnerability_array[:, 1], [1]))
        
        # Interpolate vulnerability curve values at the mean intensity levels
        interpolated_values = np.interp(mean_imls, curve_imls, curve_ordinates)
        
        # Compute the average annual loss
        return np.dot(interpolated_values, rate_occ)
