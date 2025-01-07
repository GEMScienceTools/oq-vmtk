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
                          'fitted_x': np.exp(xvec),                                  # fitted intensity measure range
                          'fitted_y': np.exp(yvec),                                  # fitted edps 
                          'intensities': intensities,                                # sampled intensities for fragility analysis
                          'poes': poes,                                              # probabilities of exceedance of each damage state (DS1 to DSi)
                          'medians': thetas,                                         # median seismic intensities (in g)
                          'betas_total': betas,                                      # associated total dispersion (accounting for building-to-building and modelling uncertainties)
                          'b1': p_cens[0],                                           # cloud analysis regression parameter (a in EDP = aIM^b)
                          'b0': p_cens[1],                                           # cloud analysis regression parameter (b in EDP = aIM^b)
                          'sigma': p_cens[2]}                                        # the standard error in the fitted regression
                                                    
        return cloud_dict


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


    def get_building_vulnerability(self,
                                   structural_loss,
                                   nonstructural_loss,
                                   occupancy_type,
                                   intensities = np.round(np.geomspace(0.05, 10.0, 50), 3),
                                   uncertainty=True):
        """
        Function to calculate the building expected loss ratio associated via a factor-based combination of
        the structural and non-structural expected losses given an occupancy type (i.e., residential, commercial 
        and industrial)
        ----------
        Parameters
        ----------
        structural_loss:                   array               Expected structural loss ratio
        nonstructural_loss:                array               Expected nonstructural loss ratio
        occupancy_type:                   string               Occupancy type (options are: RES, COM, IND)
        -------
        Returns
        -------
        building_loss:                     array               Building loss ratio
        """        
        
        if occupancy_type.lower() == 'res':
            sv  = 3.0/8.0 
            nsv = 5.0/8.0
        elif occupancy_type.lower() == 'com':
            sv = 2.0/5.0
            nsv = 3.0/5.0
        elif occupancy_type.lower() == 'ind':
            sv  = 3.0/8.0 
            nsv = 5.0/8.0
        
        loss = np.zeros((structural_loss.shape[0],1)) 
        loss = sv*structural_loss + nsv*nonstructural_loss
        
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
                
    def get_total_vulnerability(self, 
                                structural_loss, 
                                nonstructural_loss,
                                contents_loss,
                                occupancy_type,
                                intensities = np.round(np.geomspace(0.05, 10.0, 50), 3),
                                uncertainty=True):
        """
        Function to calculate the total vulnerability function associated with a building class via a factor-based combination of
        the structural, non-structural and contents vulnerability functions and for an occupancy type (i.e., residential, commercial 
        and industrial)
        ----------
        Parameters
        ----------
        structural_loss:               array                Expected structural loss ratio
        nonstructural_loss:            array                Expected non-structural loss ratio
        contents_loss:                 array                Expected contents loss ratio
        occupancy_type:               string                Occupancy type (options are: RES, COM, IND)
        -------
        Returns
        -------
        total_loss:                   array                 Total loss ratio
        """        
    
        if occupancy_type.lower() == 'res':
            sv  = 0.30 
            nsv = 0.50
            cv  = 0.20
        elif occupancy_type.lower() == 'com':
            sv  = 0.20 
            nsv = 0.30
            cv  = 0.50
        elif occupancy_type.lower() == 'ind':
            sv  = 0.15 
            nsv = 0.25
            cv  = 0.60 
            
        loss = np.zeros((structural_loss.shape[0],1))
        loss = sv*structural_loss + nsv*nonstructural_loss + cv*contents_loss

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


