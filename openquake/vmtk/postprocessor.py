import numpy as np
import pandas as pd
from scipy import stats, optimize

class postprocessor():

    """
    Details
    -------
    Class of functions to post-process results of nonlinear time-history analysis
    """
    
    def __init__(self):
        pass                
    
    def do_cloud_analysis(self,
                          imls,
                          edps,
                          damage_thresholds,
                          lower_limit,
                          censored_limit,
                          sigma_build2build=0.3,
                          intensities = np.round(np.geomspace(0.05, 10.0, 50), 3)):
        
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
        -------
        Returns
        -------
        cloud_dict:              dict         Cloud analysis outputs (regression coefficients and data, fragility parameters and functions)        
        """    
    
        # Convert to numpy array type
        if isinstance(imls, np.ndarray):
            pass
        else:
            imls = np.array(imls)
        if isinstance(edps, np.ndarray):
            pass
        else:
            edps = np.array(edps)
              
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
        
        # reproduce the fit
        xvec = np.linspace(np.log(min(imls)),np.log(max(imls)),endpoint=True)
        yvec = p_cens[0]*xvec+p_cens[1]
        
        # calculate probabilities of exceedance
        thetas = [np.exp((np.log(x)-p_cens[1])/p_cens[0]) for x in damage_thresholds] # calculate the median seismic intensities via the regression coefficients
        betas = [np.sqrt((p_cens[2]/p_cens[0])**2+sigma_build2build**2)]*4            # calculate the total uncertainty accounting for the modelling uncertainty
        poes = np.zeros((len(intensities),len(damage_thresholds)))               # initialise and calculate the probabilities of exceedances associated with each damage state
        for i in range((len(damage_thresholds))):
            poes[:,i] = self.get_fragility_function(thetas[i], betas[i])
    
        ## Package the outputs
        cloud_dict =     {'imls': imls,                                              # Input intensity measure levels
                          'edps': edps,                                              # Input engineering demand parameters
                          'lower_limit': lower_limit,                                # Input lower censoring limit
                          'upper_limit': censored_limit,                             # Input upper censoring limit
                          'damage_thresholds': damage_thresholds,                    # Input damage thresholds
                          'fitted_x': np.exp(xvec),                                  # Fitted intensity measure range
                          'fitted_y': np.exp(yvec),                                  # Fitted edps 
                          'intensities': intensities,                                # Sampled intensities for fragility analysis
                          'poes': poes,                                              # Probabilities of exceedance of each damage state (DS1 to DSi)
                          'medians': thetas,                                         # Median seismic intensities (in g)
                          'betas_total': betas,                                      # Associated total dispersion (accounting for building-to-building and modelling uncertainties)
                          'b1': p_cens[0],                                           # Cloud analysis regression parameter (a in EDP = aIM^b)
                          'b0': p_cens[1],                                           # Cloud analysis regression parameter (b in EDP = aIM^b)
                          'sigma': p_cens[2]}                                        # Standard error in the fitted regression
                                                    
        return cloud_dict



    def do_multiple_stripe_analysis(self,
                                    imls, 
                                    edps, 
                                    damage_thresholds, 
                                    sigma_build2build=0.3, 
                                    intensities = np.round(np.geomspace(0.05, 10.0, 50), 3)):
    
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
            
            eta = sol.x[0]  # Median (θ)
            beta = sol.x[1] # Dispersion (β)
            
            # Store results for each damage threshold
            results[threshold] = (eta, beta)
        
        # Calculate probabilities of exceedance for given intensity levels
        poes = np.zeros((len(intensities), len(damage_thresholds)))
        for i, threshold in enumerate(damage_thresholds):
            eta = results[threshold][0]
            beta = results[threshold][1]
            beta_total = np.sqrt(beta**2 + sigma_build2build**2)  # Total dispersion
            fitted_exceedance = stats.norm.cdf(np.log(intensities / eta) / beta_total)
            poes[:, i] = fitted_exceedance
        
        # Create the msa_dict with all relevant information
        msa_dict = {'imls': imls,                                                                                                  # Input intensity measure levels
                    'edps': edps,                                                                                                  # Input engineering demand parameters
                    'damage_thresholds': damage_thresholds,                                                                        # Input damage thresholds
                    'medians': [results[threshold][0] for threshold in damage_thresholds],                                         # Median seismic intensities (in g)
                    'betas_total': [np.sqrt(results[threshold][1]**2 + sigma_build2build**2) for threshold in damage_thresholds],  # Associated total dispersion (accounting for building-to-building and modelling uncertainties)
                    'poes': poes,                                                                                                  # Probabilities of exceedance of each damage state (DS1 to DSi)
                    'intensities': intensities}                                                                                    # Sampled intensities for fragility analysis 
        
    
        return msa_dict


    def get_fragility_function(self,
                               theta, 
                               beta_total,
                               intensities = np.round(np.geomspace(0.05, 10.0, 50), 3)):
        """
        Function to calculate the damage state lognormal CDF given median seismic intensity and associated dispersion
        ----------
        Parameters
        ----------
        intensities:                   list                Intensity measure levels 
        theta:                        float                Median seismic intensity given edp-based damage threshold.
        beta_total:                   float                Total uncertainty (i.e. accounting for record-to-record and modelling variabilities).
        -------
        Returns
        -------
        poes:                          list                Probabilities of damage exceedance.
        """
        
        ### calculate probabilities of exceedance for a range of intensity measure levels
        poes = stats.lognorm.cdf(intensities, s=beta_total, loc=0, scale=theta)            
        return poes
        
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
            df = pd.DataFrame({'IMLs': intensities,
                               'Loss': loss,
                               'COV':  cov})
            
        else:
            # Store to DataFrame
            df = pd.DataFrame({'IMLs': intensities,
                               'Loss': loss})
            
                             
        return df
        
    def calculate_sigma_loss(self, loss):
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
        
        sigma_loss_ratio = np.zeros(loss.shape)
        a_beta_dist = np.zeros(loss.shape)
        b_beta_dist = np.zeros(loss.shape)
        
        for i in range(loss.shape[0]):
            if loss[i]==0:
                  sigma_loss_ratio[i]=np.array([0])
            elif loss[i]==1:
                  sigma_loss_ratio[i]=np.array([1])
            else:
                  sigma_loss_ratio[i]=np.sqrt(loss[i]*(-0.7-2*loss[i]+np.sqrt(6.8*loss[i]+0.5)))
                
        return sigma_loss_ratio,a_beta_dist,b_beta_dist

    
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
        
        max_integration=return_period/max_return_period             
        hazard_array=hazard_array[np.where(hazard_array[:,1]>=max_integration)]
            
        mean_imls=(hazard_array[0:-1,0]+hazard_array[1:,0])/2
        rate_occ=(hazard_array[0:-1,1]/return_period)-(hazard_array[1:,1]/return_period) # Caclulate the rate of occurrence
            
        curve_imls=np.concatenate(([0],fragility_array[:,0],[20]))
        curve_ordinates=np.concatenate(([0],fragility_array[:,1],[1]))
            
        average_annual_damage_probability=np.sum(np.multiply(np.interp(mean_imls,curve_imls,curve_ordinates),rate_occ))
            
        return average_annual_damage_probability

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
        
        max_integration=return_period/max_return_period             
        hazard_array=hazard_array[np.where(hazard_array[:,1]>=max_integration)]
            
        mean_imls=(hazard_array[0:-1,0]+hazard_array[1:,0])/2
        rate_occ=(hazard_array[0:-1,1]/return_period)-(hazard_array[1:,1]/return_period) # Caclulate the rate of occurrence
            
        curve_imls=np.concatenate(([0],vulnerability_array[:,0],[20]))
        curve_ordinates=np.concatenate(([0],vulnerability_array[:,1],[1]))
            
        average_annual_loss=np.sum(np.multiply(np.interp(mean_imls,curve_imls,curve_ordinates),rate_occ))
            
        return average_annual_loss
