import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import lognorm 
from scipy import stats, optimize
from statsmodels.miscmodels.ordinal_model import OrderedModel

class postprocessor():

    """
    Details
    -------
    Class of functions to post-process results of nonlinear time-history analysis
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
                          rotation_percentile = 0.1):
        
        """
        Function to perform censored cloud analysis to a set of engineering demand parameters and intensity measure levels
        Processes cloud analysis and fits linear regression after due consideration of collapse
        ----------
        Parameters
        ----------
        imls:                    list          Intensity measure levels 
        edps:                    list          Engineering demand parameters (e.g., maximum interstorey drifts, maximum peak floor acceleration, top displacements, etc.)
        damage_thresholds        list          EDP-based damage thresholds associated with slight, moderate, extensive and complete damage
        lower_limit             float          Minimum EDP below which cloud records are filtered out (Typically equal to 0.1 times the yield capacity which is a proxy for no-damage)
        censored_limit          float          Maximum EDP above which cloud records are filtered out (Typically equal to 1.5 times the ultimate capacity which is a proxy for collapse)
        sigma_build2build       float          Building-to-building variability or modelling uncertainty (Default is set to 0.3)
        intensities             array          Array of intensity measure levels to sample the fragility functions
        fragility_rotation       bool          Bool to indicate whether or not to rotate the fragility function about a target percentile (Default set to False)
        rotation_percentile     float          Target percentile for fragility function rotation (Default set to 0.1 i.e., 10-th percentile)
        -------
        Returns
        -------
        cloud_dict:              dict         Cloud analysis outputs (regression coefficients and data, fragility parameters and functions)        
        """    

        # Convert inputs to numpy arrays
        imls = np.asarray(imls)
        edps = np.asarray(edps)
        
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
              p = np.array([stats.norm.pdf(observed[i], loc=x[1]+x[0]*x_array[i], scale=x[2]) for i in range(len(observed))],dtype=float)
              return -np.sum(np.log(p))
        sol1=optimize.fmin(func,[1,1,1],disp=False)
        
        def func2(x):
              p1 = np.array([stats.norm.pdf(observed[i], loc=x[1]+x[0]*x_array[i], scale=x[2]) for i in range(len(observed)) if bool_is_censored[i]==0],dtype=float)
              p2 = np.array([1-stats.norm.cdf(observed[i], loc=x[1]+x[0]*x_array[i], scale=x[2]) for i in range(len(observed)) if bool_is_censored[i]==1],dtype=float)
              return -np.sum(np.log(p1[p1 != 0]))-np.sum(np.log(p2[p2 != 0]))
        
        p_cens=optimize.fmin(func2,[sol1[0],sol1[1],sol1[2]],disp=False)

        # Regression fit
        xvec = np.linspace(np.log(min(imls)), np.log(max(imls)), 100)
        yvec = p_cens[0] * xvec + p_cens[1]
        
        # Compute fragility parameters
        thetas      = np.exp((np.log(damage_thresholds) - p_cens[1]) / p_cens[0]) # Median intensities
        betas_r     = np.full(len(damage_thresholds), p_cens[2] / p_cens[0])      # Record-to-record variability
        betas_total = np.full(len(damage_thresholds), np.sqrt((p_cens[2] / p_cens[0])**2 + sigma_build2build**2)) # Total dispersion
        
        # Compute probabilities of exceedance
        if fragility_rotation:
            poes = np.array([self.get_rotated_fragility_function(theta,
                                                                 rotation_percentile, 
                                                                 beta_r, 
                                                                 sigma_build2build = sigma_build2build) for theta, beta_r in zip(thetas, betas_r)]).T        
        else:            
            poes = np.array([self.get_fragility_function(theta, 
                                                         beta_r, 
                                                         sigma_build2build = sigma_build2build) for theta, beta_r in zip(thetas, betas_r)]).T
        
        cloud_dict = {
            'imls': imls, 'edps': edps, 'lower_limit': lower_limit, 'upper_limit': censored_limit,
            'damage_thresholds': damage_thresholds, 'fitted_x': np.exp(xvec), 'fitted_y': np.exp(yvec),
            'intensities': intensities, 'poes': poes, 'medians': thetas, 'betas_total': betas_total,
            'b1': p_cens[0], 'b0': p_cens[1], 'sigma': p_cens[2]
        }
        
        return cloud_dict

    def do_multiple_stripe_analysis(self,
                                    imls, 
                                    edps, 
                                    damage_thresholds, 
                                    sigma_build2build=0.3, 
                                    intensities = np.round(np.geomspace(0.05, 10.0, 50), 3),
                                    fragility_rotation = False,
                                    rotation_percentile = 0.10):
    
        """
        Function to perform maximum likelihood estimation (MLE) for fragility cuve fitting 
        ----------
        Parameters
        ----------
        imls:                    list          Intensity measure levels 
        edps:                    list          Engineering demand parameters (e.g., maximum interstorey drifts, maximum peak floor acceleration, top displacements, etc.)
        damage_thresholds        list          EDP-based damage thresholds associated with slight, moderate, extensive and complete damage
        sigma_build2build       float          Building-to-building variability or modelling uncertainty (Default is set to 0.3)
        intensities             array          Intensity measure levels to sample fragility functions from
        -------
        Returns
        -------
        msa_dict:                dict          Multiple stripe analysis outputs (medians and dispersions, and probabilities of exceedances)        
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
            beta_r = sol.x[1] # Dispersion (β) due to record-to-record variability
            
            # Store results for each damage threshold
            results[threshold] = (theta, beta_r)
        
        # Calculate probabilities of exceedance for given intensity levels
        poes = np.zeros((len(intensities), len(damage_thresholds)))
        for i, threshold in enumerate(damage_thresholds):
            theta = results[threshold][0]
            beta_r = results[threshold][1]
            if fragility_rotation:
                poes[:, i] = self.get_rotated_fragility_function(theta, 
                                                                 rotation_percentile, 
                                                                 beta_r, 
                                                                 beta_u = sigma_build2build)
            else:
                poes[:, i] = self.get_fragility_function(theta, 
                                                         beta_r, 
                                                         beta_u = sigma_build2build)
                
        
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
        Function to calculate the sigma in loss estimates based on the recommendations in:
        'Silva, V. (2019) Uncertainty and correlation in seismic vulnerability 
        functions of building classes. Earthquake Spectra. 
        DOI: 10.1193/013018eqs031m.'
        ----------
        Parameters
        ----------
        loss:                           list          Expected loss ratio
        -------
        Returns
        -------
        sigma_loss_ratio:               list          The uncertainty associated with mean loss ratio
        a_beta_dist:                   float          coefficient of the beta-distribution
        b_beta_dist:                   float          coefficient of the beta_distribution
        
        """    
        sigma_loss_ratio = np.where(loss == 0, 0,
                                    np.where(loss == 1, 1,
                                             np.sqrt(loss * (-0.7 - 2 * loss + np.sqrt(6.8 * loss + 0.5)))))
        a_beta_dist = np.zeros(loss.shape)
        b_beta_dist = np.zeros(loss.shape)
        return sigma_loss_ratio, a_beta_dist, b_beta_dist

    
    def get_rotated_fragility_function(self,
                                       theta, 
                                       percentile,
                                       beta_r, 
                                       sigma_build2build = 0.30,
                                       intensities = np.round(np.geomspace(0.05, 10.0, 50), 3)):
        """
        Function to calculate the damage state lognormal CDF given median seismic intensity and total associated dispersion
        ----------
        Reference
        ----------
        Porter (2017), When Addressing Epistemic Uncertainty in a Lognormal Fragility Function,
        How Should One Adjust the Median, Proceedings of the 16th World Conference on Earthquake
        Engineering (16WCEE), Santiago, Chile

        ----------
        Parameters
        ----------
        theta:                        float                Median seismic intensity given edp-based damage threshold.
        percentile:                   float                Target percentile for fragility function rotation (e.g., 0.2 corresponds to 20th percentile)
        beta_r:                       float                Uncertainty associated with record-to-record variability
        beta_u:                       float                Uncertainty associated with modelling variability (Default set to 0.3)
        intensities:                   list                Intensity measure levels (Default set to np.geomspace(0.05, 10.0, 50), 3))
    
        -------
        Returns
        -------
        poes:                          list                Probabilities of damage exceedance.
        """
    
        # Calculate the combined logarithmic standard deviation
        beta_total = np.sqrt(beta_r**2 + sigma_build2build**2)
        
        # Calculate the new median after rotation
        theta_prime = theta * np.exp(-stats.norm.ppf(percentile) * (beta_total - beta_r))
        
        # Calculate the rotated lognormal CDF
        return stats.lognorm(s=beta_total, scale=theta_prime).cdf(intensities)
 
        
    def get_vulnerability_function(self,
                                   poes,
                                   consequence_model,
                                   intensities = np.round(np.geomspace(0.05, 10.0, 50), 3),
                                   uncertainty = True):
        """
        Function to calculate the vulnerability function given the probabilities of exceedance and a consequence model
        ----------
        Parameters
        ----------
        poes:                         array                Probabilities of exceedance associated with the damage states considered (size = Intensity measure levels x nDS)
        consequence_model:             list                Damage-to-loss ratios       
        intensities:                  array                Intensity measure levels
        uncertainty:                   bool                Flag to calculate (or not) the coefficient of variation associated with Loss|IM
        ----------
        References
        ----------
        The coefficient of variation is calculated per:
        Silva V. Uncertainty and Correlation in Seismic Vulnerability Functions of Building Classes. Earthquake Spectra. 2019;35(4):1515-1539. doi:10.1193/013018EQS031M
        -------
        Returns
        -------
        loss:                         array                Expected loss ratios (the uncertainty is modelled separately)
        """    
        
        ### Do some consistency checks
        if len(consequence_model)!=np.size(poes,1):
              raise Exception('Mismatch between the fragility consequence models!')
        if len(intensities)!=np.size(poes,0):
              raise Exception('Mismatch between the number of IMLs and fragility models!')
        
        loss=np.zeros([len(intensities),])
        for i in range(len(intensities)):
              for j in range(0,np.size(poes,1)):
                    if j==(np.size(poes,1)-1):
                          loss[i,]=loss[i,]+poes[i,j]*consequence_model[j]
                    else:
                          loss[i,]=loss[i,]+(poes[i,j]-poes[i,j+1])*consequence_model[j]
         
        if uncertainty:            
            # Calculate the coefficient of variation assuming the Silva et al.
            cov=np.zeros(loss.shape)   
            for m in range(loss.shape[0]):                        
                mean_loss_ratio=loss[m]
                if mean_loss_ratio<1e-4:
                    loss[m]=1e-8
                    cov[m] = 1e-8
                elif np.abs(1-mean_loss_ratio)<1e-4:
                    loss[m]= 0.99999
                    cov[m] = 1e-8
                else:                                  
                    sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                    cov[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.90*max_sigma/mean_loss_ratio])
     
            # Store to DataFrame
            df = pd.DataFrame({'IML': intensities,
                               'Loss': loss,
                               'COV':  cov})
            
        else:
            # Store to DataFrame
            df = pd.DataFrame({'IML': intensities,
                               'Loss': loss})
            
                             
        return df
            
    def calculate_average_annual_damage_probability(self, 
                                                    fragility_array, 
                                                    hazard_array, 
                                                    return_period=1, 
                                                    max_return_period=5000):
        """
        Function to calculate the average annual damage state probability
        ----------
        Parameters
        ----------
        fragility_array:               array          2D array with intensity measure levels and probabilities of exceedance
                                                      as first and second columns, respectively
        hazard_array:                  array          2D array with intensity measure levels and annual rates of exceedance
                                                      as first and second columns, respectively
        -------
        Returns
        -------
        average_annual_damage_probability:     float   Average annual damage state probability
        """    
        max_integration = return_period / max_return_period
        hazard_array = hazard_array[hazard_array[:, 1] >= max_integration]
        
        mean_imls = (hazard_array[:-1, 0] + hazard_array[1:, 0]) / 2
        rate_occ = np.diff(hazard_array[:, 1]) / -return_period
        
        curve_imls = np.hstack(([0], fragility_array[:, 0], [20]))
        curve_ordinates = np.hstack(([0], fragility_array[:, 1], [1]))
        
        interpolated_values = np.interp(mean_imls, curve_imls, curve_ordinates)
        return np.dot(interpolated_values, rate_occ)
    
    def calculate_average_annual_loss(self, 
                                      vulnerability_array, 
                                      hazard_array, 
                                      return_period=1, 
                                      max_return_period=5000):
        """
        Function to calculate the average annual losses
        ----------
        Parameters
        ----------
        vulnerability_array:           array          2D array with intensity measure levels and loss ratios
                                                      as first and second columns, respectively
        hazard_array:                  array          2D array with intensity measure levels and annual rates of exceedance
                                                      as first and second columns, respectively
        -------
        Returns
        -------
        average_annual_loss:     float   Average annual damage state probability
        """    
        max_integration = return_period / max_return_period
        hazard_array = hazard_array[hazard_array[:, 1] >= max_integration]
        
        mean_imls = (hazard_array[:-1, 0] + hazard_array[1:, 0]) / 2
        rate_occ = np.diff(hazard_array[:, 1]) / -return_period
        
        curve_imls = np.hstack(([0], vulnerability_array[:, 0], [20]))
        curve_ordinates = np.hstack(([0], vulnerability_array[:, 1], [1]))
        
        interpolated_values = np.interp(mean_imls, curve_imls, curve_ordinates)
        return np.dot(interpolated_values, rate_occ)
