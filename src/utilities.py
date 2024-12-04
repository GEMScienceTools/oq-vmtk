### Import libraries
import pandas as pd
import numpy as np
import os
import re
import time
import pickle
import math
from math import sqrt, pi
from scipy import stats, optimize, signal, integrate
from scipy.interpolate import interp1d
from itertools import count

##########################################################################
#                    GENERIC UTILITY FUNCTIONS                           #
##########################################################################
def fun_lognormal(x, sigma, mu):
    """
    Function to reproduce or fit a Lognormal function to data
    -----
    Input
    -----
    :param x:                list                x-axis data
    :param sigma:           float                fitting coefficient
    :param mu:              float                fitting coefficient

    ------
    Output
    ------
    Data following Lognormal distribution
    """    
    return stats.norm.cdf(np.log(x), loc=np.log(mu), scale=sigma)

def fun_weibull(x, a, b, c):
    """
    Function to reproduce or fit a Weibull function to data
    -----
    Input
    -----
    :param x:                list                x-axis data
    :param a:               float                fitting coefficient
    :param b:               float                fitting coefficient
    :param c:               float                fitting coefficient

    ------
    Output
    ------
    Data following Weibull distribution
    """    
    return a * (1 - np.exp(-(x / b) ** c))


def fun_logit(x,a,b):
    """
    Function to reproduce or fit a Logistic function to data
    -----
    Input
    -----
    :param x:                list                x-axis data
    :param a:               float                fitting coefficient
    :param b:               float                fitting coefficient

    ------
    Output
    ------
    Data following Logistic distribution
    """    

    return np.exp(-(a+b*np.log(x)))/(1+np.exp(-(a+b*np.log(x))))

### Function to look for substrings
def find_between( s, first, last ):
    """
    Function to search for substrings
    -----
    Input
    -----
    :param x:                list                x-axis data
    :param a:               float                fitting coefficient
    :param b:               float                fitting coefficient

    ------
    Output
    ------
    Data following Logistic distribution
    """    

    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""

def import_from_pkl(path):
    """
    Function to import data stored in a pickle object
    -----
    Input
    -----
    :param path:           string                Path to the pickle file

    ------
    Output
    ------
    Pickle file
    """    

    # import file
    with open(path, 'rb') as file:
        return pickle.load(file)

def export_to_pkl(path, var):
    """
    Function to store data in a pickle object
    -----
    Input
    -----
    :param path:           string                Path to the pickle file
    :param var:          variable                Variable to store 
    ------
    Output
    ------
    Pickle file
    """    

    # store file
    with open(path, 'wb') as file:
        return pickle.dump(var, file)

### Function to read one column file
def read_one_column_file(file_name):
    """
    Function to read one column file
    -----
    Input
    -----
    :param file_name:      string                Path to the file (could be txt) including the name of the file
    ------
    Output
    ------
    x:                       list                One-column data stored in the file
    """    

    with open(file_name, 'r') as data:
        x = []
        for number in data:
            x.append(float(number))
    return x

### Function to read two-column file
def read_two_column_file(file_name):
    """
    Function to read two column file
    -----
    Input
    -----
    :param file_name:      string                Path to the file (could be txt) including the name of the file
    ------
    Output
    ------
    x:                       list                1st column of data stored in the file
    y:                       list                2nd column of data stored in the file

    """    

    with open(file_name, 'r') as data:
        x = []
        y = []
        for line in data:
            p = line.split()
            x.append(float(p[0]))
            y.append(float(p[1]))
    return x, y

### Function to sort items alphanumerically
def sorted_alphanumeric(data):
    """
    Function to sort data alphanumerically
    -----
    Input
    -----
    :param data:             list                Data to be sorted
    ------
    Output
    ------
    Sorted data of the same type as "data"
    """    
    
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


def select_files(folder=".", start="", end="", contain="", include_path=False):
    """
    Function to select files inside a folder
    -----
    Input
    -----
    :param folder:           string                Folder name, by default current one
    :param start:            string                Select the files that start with a given string 
    :param end:              string                Select the files that end with a given string 
    :param contain:          string                Select the files that contain a given string
    
    ------
    Output
    ------
    Returns a list_names of files if more than one
    """    
    files = []
    for file_name in os.listdir(folder):
        if file_name.startswith(start):
            if file_name.endswith(end):
                if isinstance(contain, str):                    
                    if file_name.find(contain) != -1:
                        if include_path==True:
                            files.append(os.path.join(folder, file_name))
                        else:
                            files.append(file_name)
                else:
                    for conts in contain:
                        if file_name.find(conts) != -1:
                            if include_path==True:
                                files.append(os.path.join(folder, file_name))
                            else:
                                files.append(file_name)
    if len(files) == 1:
        return files[0]
    else:
        assert len(files) != 0, '\nNo files selected\n'
        files.sort()
        return files

def processESMfile(in_filename, content, out_filename):
    """
    Processes acceleration history for ESM data file
    (.asc format)
    Parameters
    ----------
    in_filename : str, optional
        Location and name of the input file.
        The default is None
    content : str, optional
        Raw content of the .AT2 file.
        The default is None
    out_filename : str, optional
        location and name of the output file.
        The default is None.
    Notes
    -----
    At least one of the two variables must be defined: in_filename, content.
    Returns
    -------
    ndarray (n x 1)
        time array, same length with npts.
    ndarray (n x 1)
        acceleration array, same length with time unit
        usually in (g) unless stated otherwise.
    str
        Description of the earthquake (e.g., name, year, etc).
    """
    try:
        # Read the file content from inFilename
        if content is None:
            with open(in_filename, 'r') as file:
                content = file.readlines()
        desc = content[:64]
        dt = float(difflib.get_close_matches(
            'SAMPLING_INTERVAL_S', content)[0].split()[1])
        acc_data = content[64:]
        acc = np.asarray([float(data) for data in acc_data], dtype=float)
        dur = len(acc) * dt
        t = np.arange(0, dur, dt)
        acc = acc / 980.655  # cm/s**2 to g
        if out_filename is not None:
            np.savetxt(out_filename, acc, fmt='%1.4e')
        return t, acc, desc
    except BaseException as error:
        print(f"Record file reader FAILED for {in_filename}: ", error)

def processNGAfile(filepath, scalefactor=None):
    """
    This function process acceleration history for NGA data file (.AT2 format)
    to a single column value and return the total number of data points and 
    time iterval of the recording.
    -----
    Input
    -----
    filepath : string (location and name of the file)
    scalefactor : float (Optional) - multiplier factor that is applied to each
                  component in acceleration array.
    
    ------
    Output
    ------
    desc: Description of the earthquake (e.g., name, year, etc)
    npts: total number of recorded points (acceleration data)
    dt: time interval of recorded points
    time: array (n x 1) - time array, same length with npts
    inp_acc: array (n x 1) - acceleration array, same length with time
             unit usually in (g) unless stated as other.
    
    Example: (plot time vs acceleration)
    filepath = os.path.join(os.getcwd(),'motion_1')
    desc, npts, dt, time, inp_acc = processNGAfile (filepath)
    plt.plot(time,inp_acc)
        
    """
   
    try:
        if not scalefactor:
            scalefactor = 1.0
        with open(filepath,'r') as f:
            content = f.readlines()
        counter = 0
        desc, row4Val, acc_data = "","",[]
        for x in content:
            if counter == 1:
                desc = x
            elif counter == 3:
                row4Val = x
                if row4Val[0][0] == 'N':
                    val = row4Val.split()
                    npts = float(val[(val.index('NPTS='))+1].rstrip(','))
                    dt = float(val[(val.index('DT='))+1])
                else:
                    val = row4Val.split()
                    npts = float(val[0])
                    dt = float(val[1])
            elif counter > 3:
                data = str(x).split()
                for value in data:
                    a = float(value) * scalefactor
                    acc_data.append(a)
                inp_acc = np.asarray(acc_data)
                time = []
                for i in range (0,len(acc_data)):
                    t = i * dt
                    time.append(t)
            counter = counter + 1
        return desc, npts, dt, time, inp_acc
    except IOError:
        print("processMotion FAILED!: File is not in the directory")

    
def resample_y_values_based_on_x_values(new_x, old_x, old_y):
    """
    Function to resample data after changing x-axis values
    -----
    Input
    -----
    :param new_x:             list                New x-axis values
    :param old_x:             list                Previous x-axis values
    :param old_y:             list                Previous y-axis values

    ------
    Output
    ------
    new_y:                    list                New y-axis values

    """    
    
    f = interp1d(old_x, old_y,fill_value='extrapolate')      
    new_y = f(new_x) 
    return new_y


def RSE(y_true, y_predicted):
    """
    Function to calculate the relative squared error for uncertainty quantification in regression
    -----
    Input
    -----
    :param y_true:         list or array          Empirical data
    :param y_predicted:    list or array          Predicted data from regression

    ------
    Output
    ------
    rse:                           float          Relative squared error

    """    
 
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    RSS = np.sum(np.square(y_true - y_predicted))
    rse = math.sqrt(RSS / (len(y_true) - 2))
    return rse

def remove_elements_at_indices(test_list, idx_list):
    """
    Function to remove items from list based on their index
    -----
    Input
    -----
    :param test_list:               list          List 
    :param idx_list:                list          List of indexes of items to remove from list

    ------
    Output
    ------
    sub_list:                       list          Filtered list

    """    

    # Base case: if index list is empty, return original list
    if not idx_list:
        return test_list
    # Recursive case: extract first index and recursively process the rest of the list
    first_idx = idx_list[0]
    rest_of_indices = idx_list[1:]
    sub_list = remove_elements_at_indices(test_list, rest_of_indices)
    # Remove element at current index
    sub_list.pop(first_idx)
    return sub_list


############################################################################

############################################################################

def get_capacity_values(df,build_class):
    """
    Function to extract the SDOF capacity values from the summary file (csv)

    -----
    Input
    -----
    :param df:                 DataFrame          DataFrame containing all the properties of the building classes covered in the GEM database
    :param build_class:           string          The taxonomy associated with the building class under investigation

    ------
    Output
    ------
    sdy:                           float          Spectral displacement at yield of the SDOF system
    say:                           float          Spectral acceleration at yield of the SDOF system
    sdu:                           float          Ultimate spectral displacement of the SDOF system
    ty:                            float          Period at yield of the SDOF system

    """    
    sub_df=df[df.Building_class==build_class].reset_index(drop=True)
        
    if len(sub_df.index)==0:
        return False

    else:

        n_storeys      =   int(sub_df.Number_storeys[sub_df[sub_df.Building_class == build_class].index].values[0])       # Get the number of storeys                                            
        storey_height  = float(sub_df.Storey_height[sub_df[sub_df.Building_class == build_class].index].values[0])        # Get the typical storey height  
        total_height   = n_storeys*storey_height
        gamma_factor   = float(sub_df.Real_participation_factor[sub_df[sub_df.Building_class == build_class].index].values[0]) # Get the participation factor
                
        # ---- computes yield and elastic period ----
        type_of_period=sub_df.Type_of_period_func[0]
        a_period=sub_df.a_period_param[0]
        b_period=sub_df.b_period_param[0]

        if 'Power' in type_of_period:
            # uses a power law to estimate the yield period (T=aH^b)
            ty=a_period*(total_height**b_period)

        elif 'Poly' in type_of_period:
            # uses a polynomial function to estimate the yield period (T=aH+b)
            ty=a_period*total_height+b_period

        in_yield_drift=sub_df.Yield_drift[0] # initial yield drift
        in_ult_drift=sub_df.Ult_drift[0]     # initial ultimate drift

        yield_mult_factor=sub_df.Yield_mult_factor[0]
        ult_mult_factor=sub_df.Ult_mult_factor[0]

        end_yield_drift=in_yield_drift*yield_mult_factor # final yield drift
        end_ult_drift=in_ult_drift*ult_mult_factor # final ult drift

        sdy=(end_yield_drift*total_height)/gamma_factor
        sdu=(end_ult_drift*total_height)/gamma_factor
                        
        say=(sdy*(2*np.pi/ty)**2)/9.81

        return sdy, say, sdu, ty

def duplicate_for_drift(drifts,control_nodes):
    """
    Creates data to process box plots for peak storey drifts
    -----
    Input
    -----
    :param drifts:                  list          Peak Storey Drift Quantities
    :param control_nodes:           list          Nodes of the MDOF oscillator

    ------
    Output
    ------
    x:                              list          Box plot-ready drift values
    y:                              list          Box plot-ready control nodes values
    """    

    x = []; y = []
    for i in range(len(control_nodes)-1):
        y.extend((float(control_nodes[i]),float(control_nodes[i+1])))
        x.extend((drifts[i],drifts[i]))
    y.append(float(control_nodes[i+1]))
    x.append(0.0)
    
    return x, y


def aa_calc(frag_vul_array, hzd_array, rtP=1, max_rtP=5000):
    """
    Processes average annual losses or average annual probability of collapse
    Parameters
    ----------
    frag_vul_array - 2D array containing either the vul curve or the DS4 fragilty curve depending if AAL or AAPC is needed
    hzd_array - 2D array with the hazard curve [IMLs PoE]
    rtP - return period of the hazard curve (default 1)
    max_rtP - max hazard return period considered in integration (default 5000 years)
    Returns
    -------
    aa_value - average annual loss or average annual probability of collapse depending whether vulnerability curve or
    fragility curve is provided as input
    """
    
    max_integration=rtP/max_rtP
          
    hzd_array=hzd_array[np.where(hzd_array[:,1]>=max_integration)]
        
    mean_imls=(hzd_array[0:-1,0]+hzd_array[1:,0])/2
    rate_occ=(hzd_array[0:-1,1]/rtP)-(hzd_array[1:,1]/rtP)
        
    curve_imls=np.concatenate(([0],frag_vul_array[:,0],[20]))
    curve_ordinates=np.concatenate(([0],frag_vul_array[:,1],[1]))
        
    aa_value=np.sum(np.multiply(np.interp(mean_imls,curve_imls,curve_ordinates),rate_occ))
        
    return aa_value


def write_vulnerability_outputs_to_xml(building_classes, efficiency_df, vulnerability_in, vulnerability_out):

    # Define constants
    IMT     = ['PGA','SA(0.3s)','SA(0.6s)','SA(1.0s)']         # Intensity measure labels
    IMT_XML = ['PGA','SA(0.3)','SA(0.6)','SA(1.0)']            # Intensity measure labels for xml outputs
    OCC     = ['RES','COM','IND']                              # Occupancy Types
    COMPONENTS = ['structural','contents','nonstructural']     # Component Types
    INTENSITIES = np.round(np.geomspace(0.05, 10.0, 50), 3)    
    
    for i, currentComponent in enumerate(COMPONENTS):   
         
        # Define filename to print xml to 
        filename    = f'{vulnerability_out}/vulnerability_{currentComponent}.xml'
        
        if currentComponent == 'structural':        
            startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
                  '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
                  '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="structural">',
                  '<description>structural vulnerability model</description>']        
            endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']

        elif currentComponent == 'nonstructural':
            startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
                  '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
                  '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="nonstructural">',
                  '<description>nonstructural vulnerability model</description>']
            endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
        
        elif currentComponent == 'contents':
            startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
                  '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
                  '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="contents">',
                  '<description>contents vulnerability model</description>']
            endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
        
        outFile = open(filename, 'w')
        outFile.writelines("%s\n" % text for text in startText)

        # Loop over building classes, intensity measures and occupancy types (in that order!)
        for j, currentBuildingClass in enumerate(building_classes):
            
            for k, currentIM in enumerate(IMT):
                
                if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                    
                    for l, currentOccupancy in enumerate(OCC):
                                            
                        # Load input files
                        if currentComponent =='structural':
                            inputFile   = pd.read_csv(f'{vulnerability_in}/structural/csv/structural_vulnerability_{currentBuildingClass}_{currentIM}.csv', header=None)
                        elif currentComponent == 'contents':
                            inputFile   = pd.read_csv(f'{vulnerability_in}/contents/csv/contents_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv', header=None)
                        elif currentComponent == 'nonstructural':
                            inputFile   = pd.read_csv(f'{vulnerability_in}/nonstructural/total/csv/nonstructural_total_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv', header=None)
                        
                        # Extract relevant information
                        imls        = inputFile[0].values
                        loss_ratios = inputFile[1].values
                        imt         = IMT_XML[k]
                        uncert      = True
                        ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                        
                        # Perform the calculation to write to xml               
                        if uncert:
                            cov_loss=np.zeros(loss_ratios.shape)               
                            
                            for m in range(loss_ratios.shape[0]):
                                
                                mean_loss_ratio=loss_ratios[m]
                                
                                if mean_loss_ratio<1e-4:
                                    loss_ratios[m]=1e-8
                                    cov_loss[m] = 1e-8
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss_ratios[m]= 0.99999
                                    cov_loss[m] = 1e-8
                                else:                                  
                                    sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                    cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                                                  
                        outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                        outFile.write('<imls imt="%s" > ' % imt)
                        outFile.writelines('%g ' % iml for iml in imls)
                        outFile.write(' </imls>\n')
                        outFile.write('<meanLRs>')
                        outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                        outFile.write(' </meanLRs>\n')
                        outFile.write('<covLRs>')
                        outFile.writelines('%g ' % cv for cv in cov_loss)
                        outFile.write(' </covLRs>\n')
                        outFile.write('</vulnerabilityFunction>\n')
                
                else:
                    
                    pass
        
        # Close off the xml file 
        outFile.writelines('%s' % text for text in endText)    
        outFile.close()


def write_vulnerability_outputs_to_xml_singleIM(building_classes, IMT, vulnerability_in, vulnerability_out):

    # Define constants
    OCC     = ['RES','COM','IND']                              # Occupancy Types
    COMPONENTS = ['structural','contents','nonstructural']     # Component Types
    INTENSITIES = np.round(np.geomspace(0.05, 10.0, 50), 3)    

    for k, currentIM in enumerate(IMT):
        
        naming_suffix = currentIM.replace('s','').replace('(','').replace(')','').replace('.','')
        
        for i, currentComponent in enumerate(COMPONENTS):   
            
            # Define filename to print xml to 
            filename    = f'{vulnerability_out}/vulnerability_{currentComponent}_{naming_suffix}.xml'
            
            if currentComponent == 'structural':        
                startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
                      '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
                      '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="structural">',
                      '<description>structural vulnerability model</description>']        
                endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
    
            elif currentComponent == 'nonstructural':
                startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
                      '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
                      '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="nonstructural">',
                      '<description>nonstructural vulnerability model</description>']
                endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
            
            elif currentComponent == 'contents':
                startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
                      '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
                      '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="contents">',
                      '<description>contents vulnerability model</description>']
                endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
            
            outFile = open(filename, 'w')
            outFile.writelines("%s\n" % text for text in startText)
    
            # Loop over building classes, intensity measures and occupancy types (in that order!)
            for j, currentBuildingClass in enumerate(building_classes):
                                                                    
                        for l, currentOccupancy in enumerate(OCC):
                                                
                            # Load input files
                            if currentComponent =='structural':
                                inputFile   = pd.read_csv(f'{vulnerability_in}/structural/csv/structural_vulnerability_{currentBuildingClass}_{currentIM}.csv', header=None)
                            elif currentComponent == 'contents':
                                inputFile   = pd.read_csv(f'{vulnerability_in}/contents/csv/contents_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv', header=None)
                            elif currentComponent == 'nonstructural':
                                inputFile   = pd.read_csv(f'{vulnerability_in}/nonstructural/total/csv/nonstructural_total_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv', header=None)
                            
                            # Extract relevant information
                            imls        = inputFile[0].values
                            loss_ratios = inputFile[1].values
                            imt = currentIM.replace('s','')                        
                            uncert      = True
                            ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                            
                            # Perform the calculation to write to xml               
                            if uncert:
                                cov_loss=np.zeros(loss_ratios.shape)               
                                
                                for m in range(loss_ratios.shape[0]):
                                    
                                    mean_loss_ratio=loss_ratios[m]
                                    
                                    if mean_loss_ratio<1e-4:
                                        loss_ratios[m]=1e-8
                                        cov_loss[m] = 1e-8
                                    elif np.abs(1-mean_loss_ratio)<1e-4:
                                        loss_ratios[m]= 0.99999
                                        cov_loss[m] = 1e-8
                                    else:                                  
                                        sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                        max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                        sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                        cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                                                      
                            outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                            outFile.write('<imls imt="%s" > ' % imt)
                            outFile.writelines('%g ' % iml for iml in imls)
                            outFile.write(' </imls>\n')
                            outFile.write('<meanLRs>')
                            outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                            outFile.write(' </meanLRs>\n')
                            outFile.write('<covLRs>')
                            outFile.writelines('%g ' % cv for cv in cov_loss)
                            outFile.write(' </covLRs>\n')
                            outFile.write('</vulnerabilityFunction>\n')
                            
            # Close off the xml file 
            outFile.writelines('%s' % text for text in endText)    
            outFile.close()


def write_residents_vulnerability_to_xml(building_classes, efficiency_df, vulnerability_in, vulnerability_out):

    # Define constants
    IMT     = ['PGA','SA(0.3s)','SA(0.6s)','SA(1.0s)']         # Intensity measure labels
    IMT_XML = ['PGA','SA(0.3)','SA(0.6)','SA(1.0)']            # Intensity measure labels for xml outputs
    OCC     = ['RES','COM','IND']                                          # Occupancy Types
    INTENSITIES = np.round(np.geomspace(0.05, 10.0, 50), 3)
        
    # Define filename to print xml to 
    filename    = f'{vulnerability_out}/vulnerability_residents.xml'
    
    startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
          '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
          '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="residents">',
          '<description> DS3 damage state written as vulnerability function for number of homeless calculation </description>']        
    endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
    
    outFile = open(filename, 'w')
    outFile.writelines("%s\n" % text for text in startText)
    
    # Loop over building classes and occupancy types  
    for i, currentBuildingClass in enumerate(building_classes):   
                 
        for k, currentIM in enumerate(IMT):
                                        
            if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                    
                for l, currentOccupancy in enumerate(OCC):
                       
                    # Load structural DS3 input files
                    theta_ds3 = efficiency_df[efficiency_df.Building_class == currentBuildingClass].Median_ds3.tolist()[0]  # Get the DS3 (Extensive Damage) intensity 
                    beta_ds3  = efficiency_df[efficiency_df.Building_class == currentBuildingClass].Beta.tolist()[0]        # Get the DS3 (Extensive damage) dispersion                    
                    p_ds3     = get_fragility_function(INTENSITIES, theta_ds3, beta_ds3)                                                                                                     # Get the DS3 fragility function                                   
                    
                    imls = INTENSITIES
                    loss_ratios = p_ds3
    
                    # Define general information
                    imt         = IMT_XML[k]
                    uncert      = True
                    ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                    
                    # Perform the calculation to write to xml               
                    if uncert:
                        cov_loss=np.zeros(loss_ratios.shape)               
                        for m in range(loss_ratios.shape[0]):                        
                            mean_loss_ratio=loss_ratios[m]
                            if mean_loss_ratio<1e-4:
                                loss_ratios[m]=1e-8
                                cov_loss[m] = 1e-8
                            elif np.abs(1-mean_loss_ratio)<1e-4:
                                loss_ratios[m]= 0.99999
                                cov_loss[m] = 1e-8
                            else:                                  
                                sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                
                    outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                    outFile.write('<imls imt="%s" > ' % imt)
                    outFile.writelines('%g ' % iml for iml in imls)
                    outFile.write(' </imls>\n')
                    outFile.write('<meanLRs>')
                    outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                    outFile.write(' </meanLRs>\n')
                    outFile.write('<covLRs>')
                    outFile.writelines('%g ' % cv for cv in cov_loss)
                    outFile.write(' </covLRs>\n')
                    outFile.write('</vulnerabilityFunction>\n')
            
            else:
                
                pass
                       
    # Close off the xml file 
    outFile.writelines('%s' % text for text in endText)    
    outFile.close()
    
    
def write_residents_vulnerability_to_xml_singleIM(building_classes, IMT, summary_df, vulnerability_in, vulnerability_out):

    # Define constants
    OCC     = ['RES','COM','IND']                                          # Occupancy Types
    INTENSITIES = np.round(np.geomspace(0.05, 10.0, 50), 3)

    for k, currentIM in enumerate(IMT):
        
        naming_suffix = currentIM.replace('s','').replace('(','').replace(')','').replace('.','')
            
        # Define filename to print xml to 
        filename    = f'{vulnerability_out}/vulnerability_residents_{naming_suffix}.xml'
        
        startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
              '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
              '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="residents">',
              '<description> DS3 damage state written as vulnerability function for number of homeless calculation </description>']        
        endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
        
        outFile = open(filename, 'w')
        outFile.writelines("%s\n" % text for text in startText)
        
        # Loop over building classes and occupancy types  
        for i, currentBuildingClass in enumerate(building_classes):   
                                                         
                    for l, currentOccupancy in enumerate(OCC):
                           
                        # Load structural DS3 input files
                        theta_ds3 = summary_df[summary_df.Building_class == currentBuildingClass].Median_ds3.tolist()[0]  # Get the DS3 (Extensive Damage) intensity 
                        beta_ds3  = summary_df[summary_df.Building_class == currentBuildingClass].Beta.tolist()[0]        # Get the DS3 (Extensive damage) dispersion                    
                        p_ds3     = get_fragility_function(INTENSITIES, theta_ds3, beta_ds3)                                                                                                     # Get the DS3 fragility function                                   
                        
                        imls = INTENSITIES
                        loss_ratios = p_ds3
        
                        # Define general information
                        imt = currentIM.replace('s','')  
                        uncert      = True
                        ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                        
                        # Perform the calculation to write to xml               
                        if uncert:
                            cov_loss=np.zeros(loss_ratios.shape)               
                            for m in range(loss_ratios.shape[0]):                        
                                mean_loss_ratio=loss_ratios[m]
                                if mean_loss_ratio<1e-4:
                                    loss_ratios[m]=1e-8
                                    cov_loss[m] = 1e-8
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss_ratios[m]= 0.99999
                                    cov_loss[m] = 1e-8
                                else:                                  
                                    sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                    cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                    
                        outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                        outFile.write('<imls imt="%s" > ' % imt)
                        outFile.writelines('%g ' % iml for iml in imls)
                        outFile.write(' </imls>\n')
                        outFile.write('<meanLRs>')
                        outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                        outFile.write(' </meanLRs>\n')
                        outFile.write('<covLRs>')
                        outFile.writelines('%g ' % cv for cv in cov_loss)
                        outFile.write(' </covLRs>\n')
                        outFile.write('</vulnerabilityFunction>\n')
                        
        # Close off the xml file 
        outFile.writelines('%s' % text for text in endText)    
        outFile.close()


def write_area_vulnerability_to_xml(building_classes, efficiency_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    IMT     = ['PGA','SA(0.3s)','SA(0.6s)','SA(1.0s)']         # Intensity measure labels
    IMT_XML = ['PGA','SA(0.3)','SA(0.6)','SA(1.0)']            # Intensity measure labels for xml outputs
    OCC     = ['RES','COM','IND']                                          # Occupancy Types
    INTENSITIES = np.round(np.geomspace(0.05, 10.0, 50), 3)
    
    # Define filename to print xml to 
    filename    = f'{vulnerability_out}/vulnerability_area.xml'
    
    startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
          '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
          '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="area">',
          '<description> Complete damage state written as vulnerability used for area loss calculation</description>']        
    endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
    
    outFile = open(filename, 'w')
    outFile.writelines("%s\n" % text for text in startText)
    
    # Loop over building classes and occupancy types  
    for i, currentBuildingClass in enumerate(building_classes):   
                 
        for k, currentIM in enumerate(IMT):
                                        
            if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                    
                for l, currentOccupancy in enumerate(OCC):
                       
                    # Load structural DS4 input files
                    theta_ds4 = efficiency_df[efficiency_df.Building_class == currentBuildingClass].Median_ds4.tolist()[0]  # Get the DS4 (Complete Damage) intensity 
                    beta_ds4  = efficiency_df[efficiency_df.Building_class == currentBuildingClass].Beta.tolist()[0]        # Get the DS4 (Complete damage) dispersion                    
                    p_ds4     = get_fragility_function(INTENSITIES, theta_ds4, beta_ds4)                                                                                                     # Get the DS3 fragility function                                   
                    
                    imls = INTENSITIES
                    loss_ratios = p_ds4
    
                    # Define general information
                    imt         = IMT_XML[k]
                    uncert      = True
                    ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                    
                    # Perform the calculation to write to xml               
                    if uncert:
                        cov_loss=np.zeros(loss_ratios.shape)               
                        for m in range(loss_ratios.shape[0]):                        
                            mean_loss_ratio=loss_ratios[m]
                            if mean_loss_ratio<1e-4:
                                loss_ratios[m]=1e-8
                                cov_loss[m] = 1e-8
                            elif np.abs(1-mean_loss_ratio)<1e-4:
                                loss_ratios[m]= 0.99999
                                cov_loss[m] = 1e-8
                            else:                                  
                                sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                
                    outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                    outFile.write('<imls imt="%s" > ' % imt)
                    outFile.writelines('%g ' % iml for iml in imls)
                    outFile.write(' </imls>\n')
                    outFile.write('<meanLRs>')
                    outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                    outFile.write(' </meanLRs>\n')
                    outFile.write('<covLRs>')
                    outFile.writelines('%g ' % cv for cv in cov_loss)
                    outFile.write(' </covLRs>\n')
                    outFile.write('</vulnerabilityFunction>\n')
            
            else:
                
                pass
                       
    # Close off the xml file 
    outFile.writelines('%s' % text for text in endText)    
    outFile.close()
    
    
def write_area_vulnerability_to_xml_singleIM(building_classes, IMT, summary_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    OCC     = ['RES','COM','IND']                                          # Occupancy Types
    INTENSITIES = np.round(np.geomspace(0.05, 10.0, 50), 3)


    for k, currentIM in enumerate(IMT):
        
        naming_suffix = currentIM.replace('s','').replace('(','').replace(')','').replace('.','')
        
        # Define filename to print xml to 
        filename    = f'{vulnerability_out}/vulnerability_area_{naming_suffix}.xml'
        
        startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
              '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
              '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="area">',
              '<description> Complete damage state written as vulnerability used for area loss calculation</description>']        
        endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
        
        outFile = open(filename, 'w')
        outFile.writelines("%s\n" % text for text in startText)
        
        # Loop over building classes and occupancy types  
        for i, currentBuildingClass in enumerate(building_classes):   
                                                         
                    for l, currentOccupancy in enumerate(OCC):
                           
                        # Load structural DS4 input files
                        theta_ds4 = summary_df[summary_df.Building_class == currentBuildingClass].Median_ds4.tolist()[0]  # Get the DS4 (Complete Damage) intensity 
                        beta_ds4  = summary_df[summary_df.Building_class == currentBuildingClass].Beta.tolist()[0]        # Get the DS4 (Complete damage) dispersion                    
                        p_ds4     = get_fragility_function(INTENSITIES, theta_ds4, beta_ds4)                                                                                                     # Get the DS3 fragility function                                   
                        
                        imls = INTENSITIES
                        loss_ratios = p_ds4
        
                        # Define general information
                        imt = currentIM.replace('s','')
                        uncert      = True
                        ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                        
                        # Perform the calculation to write to xml               
                        if uncert:
                            cov_loss=np.zeros(loss_ratios.shape)               
                            for m in range(loss_ratios.shape[0]):                        
                                mean_loss_ratio=loss_ratios[m]
                                if mean_loss_ratio<1e-4:
                                    loss_ratios[m]=1e-8
                                    cov_loss[m] = 1e-8
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss_ratios[m]= 0.99999
                                    cov_loss[m] = 1e-8
                                else:                                  
                                    sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                    cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                    
                        outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                        outFile.write('<imls imt="%s" > ' % imt)
                        outFile.writelines('%g ' % iml for iml in imls)
                        outFile.write(' </imls>\n')
                        outFile.write('<meanLRs>')
                        outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                        outFile.write(' </meanLRs>\n')
                        outFile.write('<covLRs>')
                        outFile.writelines('%g ' % cv for cv in cov_loss)
                        outFile.write(' </covLRs>\n')
                        outFile.write('</vulnerabilityFunction>\n')
                                       
        # Close off the xml file 
        outFile.writelines('%s' % text for text in endText)    
        outFile.close()


def write_number_vulnerability_to_xml(building_classes, efficiency_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    IMT     = ['PGA','SA(0.3s)','SA(0.6s)','SA(1.0s)']         # Intensity measure labels
    IMT_XML = ['PGA','SA(0.3)','SA(0.6)','SA(1.0)']            # Intensity measure labels for xml outputs
    OCC     = ['RES','COM','IND']                                          # Occupancy Types
    INTENSITIES = np.round(np.geomspace(0.05, 10.0, 50), 3)
    
    # Define filename to print xml to 
    filename    = f'{vulnerability_out}/vulnerability_number.xml'
    
    startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
          '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
          '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="number">',
          '<description> Complete damage state written as vulnerability used for number of damaged buildings calculation</description>']        
    endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
    
    outFile = open(filename, 'w')
    outFile.writelines("%s\n" % text for text in startText)
    
    # Loop over building classes and occupancy types  
    for i, currentBuildingClass in enumerate(building_classes):   
                 
        for k, currentIM in enumerate(IMT):
                                        
            if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                    
                for l, currentOccupancy in enumerate(OCC):
                       
                    # Load structural DS4 input files
                    theta_ds4 = efficiency_df[efficiency_df.Building_class == currentBuildingClass].Median_ds4.tolist()[0]  # Get the DS4 (Complete Damage) intensity 
                    beta_ds4  = efficiency_df[efficiency_df.Building_class == currentBuildingClass].Beta.tolist()[0]        # Get the DS4 (Complete damage) dispersion                    
                    p_ds4     = get_fragility_function(INTENSITIES, theta_ds4, beta_ds4)                                                                                                     # Get the DS3 fragility function                                   
                    
                    imls = INTENSITIES
                    loss_ratios = p_ds4
    
                    # Define general information
                    imt         = IMT_XML[k]
                    uncert      = True
                    ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                    
                    # Perform the calculation to write to xml               
                    if uncert:
                        cov_loss=np.zeros(loss_ratios.shape)               
                        for m in range(loss_ratios.shape[0]):                        
                            mean_loss_ratio=loss_ratios[m]
                            if mean_loss_ratio<1e-4:
                                loss_ratios[m]=1e-8
                                cov_loss[m] = 1e-8
                            elif np.abs(1-mean_loss_ratio)<1e-4:
                                loss_ratios[m]= 0.99999
                                cov_loss[m] = 1e-8
                            else:                                  
                                sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                
                    outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                    outFile.write('<imls imt="%s" > ' % imt)
                    outFile.writelines('%g ' % iml for iml in imls)
                    outFile.write(' </imls>\n')
                    outFile.write('<meanLRs>')
                    outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                    outFile.write(' </meanLRs>\n')
                    outFile.write('<covLRs>')
                    outFile.writelines('%g ' % cv for cv in cov_loss)
                    outFile.write(' </covLRs>\n')
                    outFile.write('</vulnerabilityFunction>\n')
            
            else:
                
                pass
                       
    # Close off the xml file 
    outFile.writelines('%s' % text for text in endText)    
    outFile.close()

def write_number_vulnerability_to_xml_singleIM(building_classes, IMT, summary_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    OCC     = ['RES','COM','IND']                                          # Occupancy Types
    INTENSITIES = np.round(np.geomspace(0.05, 10.0, 50), 3)

    for k, currentIM in enumerate(IMT):
        
        naming_suffix = currentIM.replace('s','').replace('(','').replace(')','').replace('.','')
        
        # Define filename to print xml to 
        filename    = f'{vulnerability_out}/vulnerability_number_{naming_suffix}.xml'
        
        startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
              '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
              '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="number">',
              '<description> Complete damage state written as vulnerability used for number of damaged buildings calculation</description>']        
        endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
        
        outFile = open(filename, 'w')
        outFile.writelines("%s\n" % text for text in startText)
        
        # Loop over building classes and occupancy types  
        for i, currentBuildingClass in enumerate(building_classes):   
                                                         
                    for l, currentOccupancy in enumerate(OCC):
                           
                        # Load structural DS4 input files
                        theta_ds4 = summary_df[summary_df.Building_class == currentBuildingClass].Median_ds4.tolist()[0]  # Get the DS4 (Complete Damage) intensity 
                        beta_ds4  = summary_df[summary_df.Building_class == currentBuildingClass].Beta.tolist()[0]        # Get the DS4 (Complete damage) dispersion                    
                        p_ds4     = get_fragility_function(INTENSITIES, theta_ds4, beta_ds4)                                                                                                     # Get the DS3 fragility function                                   
                        
                        imls = INTENSITIES
                        loss_ratios = p_ds4
        
                        # Define general information
                        imt         = currentIM.replace('s','')
                        uncert      = True
                        ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                        
                        # Perform the calculation to write to xml               
                        if uncert:
                            cov_loss=np.zeros(loss_ratios.shape)               
                            for m in range(loss_ratios.shape[0]):                        
                                mean_loss_ratio=loss_ratios[m]
                                if mean_loss_ratio<1e-4:
                                    loss_ratios[m]=1e-8
                                    cov_loss[m] = 1e-8
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss_ratios[m]= 0.99999
                                    cov_loss[m] = 1e-8
                                else:                                  
                                    sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                    cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                    
                        outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                        outFile.write('<imls imt="%s" > ' % imt)
                        outFile.writelines('%g ' % iml for iml in imls)
                        outFile.write(' </imls>\n')
                        outFile.write('<meanLRs>')
                        outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                        outFile.write(' </meanLRs>\n')
                        outFile.write('<covLRs>')
                        outFile.writelines('%g ' % cv for cv in cov_loss)
                        outFile.write(' </covLRs>\n')
                        outFile.write('</vulnerabilityFunction>\n')
                                       
        # Close off the xml file 
        outFile.writelines('%s' % text for text in endText)    
        outFile.close()

def write_building_vulnerability_to_xml(building_classes, efficiency_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    IMT     = ['PGA','SA(0.3s)','SA(0.6s)','SA(1.0s)']         # Intensity measure labels
    IMT_XML = ['PGA','SA(0.3)','SA(0.6)','SA(1.0)']            # Intensity measure labels for xml outputs
    OCC     = ['RES','COM','IND']                                          # Occupancy Types

    # Define filename to print xml to 
    filename    = f'{vulnerability_out}/vulnerability_building.xml'
    
    startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
          '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
          '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="structural">',
          '<description>building vulnerability model</description>']        
    endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
    
    outFile = open(filename, 'w')
    outFile.writelines("%s\n" % text for text in startText)
    
    # Loop over building classes and occupancy types  
    for i, currentBuildingClass in enumerate(building_classes):   
                      
        for k, currentIM in enumerate(IMT):
                                            
            if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                    
                for l, currentOccupancy in enumerate(OCC):
                                             
                    # Load vulnerability input files
                    inputFile_struct      = pd.read_csv(f'{vulnerability_in}/structural/csv/structural_vulnerability_{currentBuildingClass}_{currentIM}.csv')
                    inputFile_nonStruct   = pd.read_csv(f'{vulnerability_in}/nonstructural/total/csv/nonstructural_total_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv')
                    inputFile_cont        = pd.read_csv(f'{vulnerability_in}/contents/csv/contents_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv')
                    
                    # Extract relevant information for structural vulnerability
                    imls_struct        = inputFile_struct[f'{inputFile_struct.columns[0]}'].values                     
                    loss_ratios_struct = inputFile_struct[f'{inputFile_struct.columns[1]}'].values
                    
                    # Extract relevant information for non-structural vulnerability
                    imls_nonStruct        = inputFile_nonStruct[f'{inputFile_nonStruct.columns[0]}'].values
                    loss_ratios_nonStruct = inputFile_nonStruct[f'{inputFile_nonStruct.columns[1]}'].values
        
                    # Define general information
                    imt         = IMT_XML[k]
                    uncert      = True
                    ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'

                    # Calculate total loss ratio
                    if currentOccupancy == 'RES':
                        sv  = 3.0/8.0 
                        nsv = 5.0/8.0
                    elif currentOccupancy == 'COM':
                        sv = 2.0/5.0
                        nsv = 3.0/5.0
                    elif currentOccupancy == 'IND':
                        sv  = 3.0/8.0 
                        nsv = 5.0/8.0
                    
                    loss_ratios = np.zeros((loss_ratios_struct.shape[0],1)) 
                    loss_ratios = sv*loss_ratios_struct + nsv*loss_ratios_nonStruct

                    # Perform the calculation to write to xml               
                    if uncert:
                        cov_loss=np.zeros(loss_ratios.shape)               
                        for m in range(loss_ratios.shape[0]):                        
                            mean_loss_ratio=loss_ratios[m]
                            if mean_loss_ratio<1e-4:
                                loss_ratios[m]=1e-8
                                cov_loss[m] = 1e-8
                            elif np.abs(1-mean_loss_ratio)<1e-4:
                                loss_ratios[m]= 0.99999
                                cov_loss[m] = 1e-8
                            else:                                  
                                sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                
                    outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                    outFile.write('<imls imt="%s" > ' % imt)
                    outFile.writelines('%g ' % iml for iml in imls_struct)
                    outFile.write(' </imls>\n')
                    outFile.write('<meanLRs>')
                    outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                    outFile.write(' </meanLRs>\n')
                    outFile.write('<covLRs>')
                    outFile.writelines('%g ' % cv for cv in cov_loss)
                    outFile.write(' </covLRs>\n')
                    outFile.write('</vulnerabilityFunction>\n')
            
            else:
                
                pass
                       
    # Close off the xml file 
    outFile.writelines('%s' % text for text in endText)    
    outFile.close()


def write_building_vulnerability_to_xml_singleIM(building_classes, IMT, summary_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    OCC     = ['RES','COM','IND']                                          # Occupancy Types

    for k, currentIM in enumerate(IMT):
        
        naming_suffix = currentIM.replace('s','').replace('(','').replace(')','').replace('.','')
    
        # Define filename to print xml to 
        filename    = f'{vulnerability_out}/vulnerability_building_{naming_suffix}.xml'
        
        startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
              '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
              '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="structural">',
              '<description>building vulnerability model</description>']        
        endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
        
        outFile = open(filename, 'w')
        outFile.writelines("%s\n" % text for text in startText)
        
        # Loop over building classes and occupancy types  
        for i, currentBuildingClass in enumerate(building_classes):   
                                                              
                    for l, currentOccupancy in enumerate(OCC):
                                                 
                        # Load vulnerability input files
                        inputFile_struct      = pd.read_csv(f'{vulnerability_in}/structural/csv/structural_vulnerability_{currentBuildingClass}_{currentIM}.csv')
                        inputFile_nonStruct   = pd.read_csv(f'{vulnerability_in}/nonstructural/total/csv/nonstructural_total_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv')
                        inputFile_cont        = pd.read_csv(f'{vulnerability_in}/contents/csv/contents_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv')
                        
                        # Extract relevant information for structural vulnerability
                        imls_struct        = inputFile_struct[f'{inputFile_struct.columns[0]}'].values                     
                        loss_ratios_struct = inputFile_struct[f'{inputFile_struct.columns[1]}'].values
                        
                        # Extract relevant information for non-structural vulnerability
                        imls_nonStruct        = inputFile_nonStruct[f'{inputFile_nonStruct.columns[0]}'].values
                        loss_ratios_nonStruct = inputFile_nonStruct[f'{inputFile_nonStruct.columns[1]}'].values
            
                        # Define general information
                        imt         = currentIM.replace('s','')
                        uncert      = True
                        ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
    
                        # Calculate total loss ratio
                        if currentOccupancy == 'RES':
                            sv  = 3.0/8.0 
                            nsv = 5.0/8.0
                        elif currentOccupancy == 'COM':
                            sv = 2.0/5.0
                            nsv = 3.0/5.0
                        elif currentOccupancy == 'IND':
                            sv  = 3.0/8.0 
                            nsv = 5.0/8.0
                        
                        loss_ratios = np.zeros((loss_ratios_struct.shape[0],1)) 
                        loss_ratios = sv*loss_ratios_struct + nsv*loss_ratios_nonStruct
    
                        # Perform the calculation to write to xml               
                        if uncert:
                            cov_loss=np.zeros(loss_ratios.shape)               
                            for m in range(loss_ratios.shape[0]):                        
                                mean_loss_ratio=loss_ratios[m]
                                if mean_loss_ratio<1e-4:
                                    loss_ratios[m]=1e-8
                                    cov_loss[m] = 1e-8
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss_ratios[m]= 0.99999
                                    cov_loss[m] = 1e-8
                                else:                                  
                                    sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                    cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                    
                        outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                        outFile.write('<imls imt="%s" > ' % imt)
                        outFile.writelines('%g ' % iml for iml in imls_struct)
                        outFile.write(' </imls>\n')
                        outFile.write('<meanLRs>')
                        outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                        outFile.write(' </meanLRs>\n')
                        outFile.write('<covLRs>')
                        outFile.writelines('%g ' % cv for cv in cov_loss)
                        outFile.write(' </covLRs>\n')
                        outFile.write('</vulnerabilityFunction>\n')
                                       
        # Close off the xml file 
        outFile.writelines('%s' % text for text in endText)    
        outFile.close()


def write_total_vulnerability_to_xml(building_classes, efficiency_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    IMT     = ['PGA','SA(0.3s)','SA(0.6s)','SA(1.0s)']         # Intensity measure labels
    IMT_XML = ['PGA','SA(0.3)','SA(0.6)','SA(1.0)']            # Intensity measure labels for xml outputs
    OCC     = ['RES','COM','IND']                                          # Occupancy Types

    # Define filename to print xml to 
    filename    = f'{vulnerability_out}/vulnerability_total.xml'
    
    startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
          '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
          '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="structural">',
          '<description>total vulnerability model</description>']        
    endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
    
    outFile = open(filename, 'w')
    outFile.writelines("%s\n" % text for text in startText)
    
    # Loop over building classes and occupancy types  
    for i, currentBuildingClass in enumerate(building_classes):   
                      
        for k, currentIM in enumerate(IMT):
                                            
            if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                    
                for l, currentOccupancy in enumerate(OCC):
                         
                    # Load vulnerability input files
                    inputFile_struct      = pd.read_csv(f'{vulnerability_in}/structural/csv/structural_vulnerability_{currentBuildingClass}_{currentIM}.csv')
                    inputFile_nonStruct   = pd.read_csv(f'{vulnerability_in}/nonstructural/total/csv/nonstructural_total_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv')
                    inputFile_cont        = pd.read_csv(f'{vulnerability_in}/contents/csv/contents_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv')
                    
                    # Extract relevant information for structural vulnerability
                    imls_struct        = inputFile_struct[f'{inputFile_struct.columns[0]}'].values                     
                    loss_ratios_struct = inputFile_struct[f'{inputFile_struct.columns[1]}'].values
                    
                    # Extract relevant information for non-structural vulnerability
                    imls_nonStruct        = inputFile_nonStruct[f'{inputFile_nonStruct.columns[0]}'].values
                    loss_ratios_nonStruct = inputFile_nonStruct[f'{inputFile_nonStruct.columns[1]}'].values

                    # Extract relevant information for contents vulnerability
                    imls_cont        = inputFile_cont[f'{inputFile_cont.columns[0]}'].values
                    loss_ratios_cont = inputFile_cont[f'{inputFile_cont.columns[1]}'].values
        
                    # Define general information
                    imt         = IMT_XML[k]
                    uncert      = True
                    ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                    
                    # Calculate total loss ratio
                    if currentOccupancy == 'RES':
                        sv  = 0.30 
                        nsv = 0.50
                        cv  = 0.20
                    elif currentOccupancy == 'COM':
                        sv  = 0.20 
                        nsv = 0.30
                        cv  = 0.50
                    elif currentOccupancy == 'IND':
                        sv  = 0.15 
                        nsv = 0.25
                        cv  = 0.60 
                                        
                    loss_ratios = np.zeros((loss_ratios_struct.shape[0],1)) 
                    loss_ratios = sv*loss_ratios_struct + nsv*loss_ratios_nonStruct + cv*loss_ratios_cont
    
                    # Perform the calculation to write to xml               
                    if uncert:
                        cov_loss=np.zeros(loss_ratios.shape)               
                        for m in range(loss_ratios.shape[0]):                        
                            mean_loss_ratio=loss_ratios[m]
                            if mean_loss_ratio<1e-4:
                                loss_ratios[m]=1e-8
                                cov_loss[m] = 1e-8
                            elif np.abs(1-mean_loss_ratio)<1e-4:
                                loss_ratios[m]= 0.99999
                                cov_loss[m] = 1e-8
                            else:                                  
                                sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                
                    outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                    outFile.write('<imls imt="%s" > ' % imt)
                    outFile.writelines('%g ' % iml for iml in imls_struct)
                    outFile.write(' </imls>\n')
                    outFile.write('<meanLRs>')
                    outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                    outFile.write(' </meanLRs>\n')
                    outFile.write('<covLRs>')
                    outFile.writelines('%g ' % cv for cv in cov_loss)
                    outFile.write(' </covLRs>\n')
                    outFile.write('</vulnerabilityFunction>\n')
            
            else:
                
                pass
                       
    # Close off the xml file 
    outFile.writelines('%s' % text for text in endText)    
    outFile.close()

def write_total_vulnerability_to_xml_singleIM(building_classes, IMT, summary_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    OCC     = ['RES','COM','IND']                                          # Occupancy Types

    for k, currentIM in enumerate(IMT):
        
        naming_suffix = currentIM.replace('s','').replace('(','').replace(')','').replace('.','')
    
        # Define filename to print xml to 
        filename    = f'{vulnerability_out}/vulnerability_total_{naming_suffix}.xml'
        
        startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
              '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
              '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="structural">',
              '<description>total vulnerability model</description>']        
        endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
        
        outFile = open(filename, 'w')
        outFile.writelines("%s\n" % text for text in startText)
        
        # Loop over building classes and occupancy types  
        for i, currentBuildingClass in enumerate(building_classes):   
                                                              
                    for l, currentOccupancy in enumerate(OCC):
                             
                        # Load vulnerability input files
                        inputFile_struct      = pd.read_csv(f'{vulnerability_in}/structural/csv/structural_vulnerability_{currentBuildingClass}_{currentIM}.csv')
                        inputFile_nonStruct   = pd.read_csv(f'{vulnerability_in}/nonstructural/total/csv/nonstructural_total_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv')
                        inputFile_cont        = pd.read_csv(f'{vulnerability_in}/contents/csv/contents_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv')
                        
                        # Extract relevant information for structural vulnerability
                        imls_struct        = inputFile_struct[f'{inputFile_struct.columns[0]}'].values                     
                        loss_ratios_struct = inputFile_struct[f'{inputFile_struct.columns[1]}'].values
                        
                        # Extract relevant information for non-structural vulnerability
                        imls_nonStruct        = inputFile_nonStruct[f'{inputFile_nonStruct.columns[0]}'].values
                        loss_ratios_nonStruct = inputFile_nonStruct[f'{inputFile_nonStruct.columns[1]}'].values
    
                        # Extract relevant information for contents vulnerability
                        imls_cont        = inputFile_cont[f'{inputFile_cont.columns[0]}'].values
                        loss_ratios_cont = inputFile_cont[f'{inputFile_cont.columns[1]}'].values
            
                        # Define general information
                        imt         = currentIM.replace('s','')
                        uncert      = True
                        ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                        
                        # Calculate total loss ratio
                        if currentOccupancy == 'RES':
                            sv  = 0.30 
                            nsv = 0.50
                            cv  = 0.20
                        elif currentOccupancy == 'COM':
                            sv  = 0.20 
                            nsv = 0.30
                            cv  = 0.50
                        elif currentOccupancy == 'IND':
                            sv  = 0.15 
                            nsv = 0.25
                            cv  = 0.60 
                                            
                        loss_ratios = np.zeros((loss_ratios_struct.shape[0],1)) 
                        loss_ratios = sv*loss_ratios_struct + nsv*loss_ratios_nonStruct + cv*loss_ratios_cont
        
                        # Perform the calculation to write to xml               
                        if uncert:
                            cov_loss=np.zeros(loss_ratios.shape)               
                            for m in range(loss_ratios.shape[0]):                        
                                mean_loss_ratio=loss_ratios[m]
                                if mean_loss_ratio<1e-4:
                                    loss_ratios[m]=1e-8
                                    cov_loss[m] = 1e-8
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss_ratios[m]= 0.99999
                                    cov_loss[m] = 1e-8
                                else:                                  
                                    sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                    cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                    
                        outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                        outFile.write('<imls imt="%s" > ' % imt)
                        outFile.writelines('%g ' % iml for iml in imls_struct)
                        outFile.write(' </imls>\n')
                        outFile.write('<meanLRs>')
                        outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                        outFile.write(' </meanLRs>\n')
                        outFile.write('<covLRs>')
                        outFile.writelines('%g ' % cv for cv in cov_loss)
                        outFile.write(' </covLRs>\n')
                        outFile.write('</vulnerabilityFunction>\n')
                                       
        # Close off the xml file 
        outFile.writelines('%s' % text for text in endText)    
        outFile.close()

    
def write_structural_vulnerability_to_xml(building_classes, efficiency_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    IMT     = ['PGA','SA(0.3s)','SA(0.6s)','SA(1.0s)']         # Intensity measure labels
    IMT_XML = ['PGA','SA(0.3)','SA(0.6)','SA(1.0)']            # Intensity measure labels for xml outputs
    OCC     = ['RES','COM','IND']                                          # Occupancy Types

    # Define filename to print xml to 
    filename    = f'{vulnerability_out}/vulnerability_structural.xml'
    
    startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
          '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
          '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="structural">',
          '<description>structural vulnerability model</description>']        
    endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
    
    outFile = open(filename, 'w')
    outFile.writelines("%s\n" % text for text in startText)
    
    # Loop over building classes and occupancy types  
    for i, currentBuildingClass in enumerate(building_classes):   
                      
        for k, currentIM in enumerate(IMT):
                                            
            if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                    
                for l, currentOccupancy in enumerate(OCC):
                                             
                    # Load structural vulnerability input files
                    inputFile_struct      = pd.read_csv(f'{vulnerability_in}/structural/csv/structural_vulnerability_{currentBuildingClass}_{currentIM}.csv')
                    
                    # Extract relevant information for structural vulnerability
                    imls_struct        = inputFile_struct[f'{inputFile_struct.columns[0]}'].values                     
                    loss_ratios_struct = inputFile_struct[f'{inputFile_struct.columns[1]}'].values
                            
                    # Define general information
                    imt         = IMT_XML[k]
                    uncert      = True
                    ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                    
                    loss_ratios = np.zeros((loss_ratios_struct.shape[0],1)) 
                    loss_ratios = loss_ratios_struct

                    # Perform the calculation to write to xml               
                    if uncert:
                        cov_loss=np.zeros(loss_ratios.shape)               
                        for m in range(loss_ratios.shape[0]):                        
                            mean_loss_ratio=loss_ratios[m]
                            if mean_loss_ratio<1e-4:
                                loss_ratios[m]=1e-8
                                cov_loss[m] = 1e-8
                            elif np.abs(1-mean_loss_ratio)<1e-4:
                                loss_ratios[m]= 0.99999
                                cov_loss[m] = 1e-8
                            else:                                  
                                sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                
                    outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                    outFile.write('<imls imt="%s" > ' % imt)
                    outFile.writelines('%g ' % iml for iml in imls_struct)
                    outFile.write(' </imls>\n')
                    outFile.write('<meanLRs>')
                    outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                    outFile.write(' </meanLRs>\n')
                    outFile.write('<covLRs>')
                    outFile.writelines('%g ' % cv for cv in cov_loss)
                    outFile.write(' </covLRs>\n')
                    outFile.write('</vulnerabilityFunction>\n')
            
            else:
                
                pass
                       
    # Close off the xml file 
    outFile.writelines('%s' % text for text in endText)    
    outFile.close()

def write_structural_vulnerability_to_xml_singleIM(building_classes, IMT, summary_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    OCC     = ['RES','COM','IND']                                          # Occupancy Types

    for k, currentIM in enumerate(IMT):
        
        naming_suffix = currentIM.replace('s','').replace('(','').replace(')','').replace('.','')
    
        # Define filename to print xml to 
        filename    = f'{vulnerability_out}/vulnerability_structural_{naming_suffix}.xml'
        
        startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
              '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
              '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="structural">',
              '<description>structural vulnerability model</description>']        
        endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
        
        outFile = open(filename, 'w')
        outFile.writelines("%s\n" % text for text in startText)
        
        # Loop over building classes and occupancy types  
        for i, currentBuildingClass in enumerate(building_classes):   
                          
                    for l, currentOccupancy in enumerate(OCC):
                                                 
                        # Load structural vulnerability input files
                        inputFile_struct      = pd.read_csv(f'{vulnerability_in}/structural/csv/structural_vulnerability_{currentBuildingClass}_{currentIM}.csv')
                        
                        # Extract relevant information for structural vulnerability
                        imls_struct        = inputFile_struct[f'{inputFile_struct.columns[0]}'].values                     
                        loss_ratios_struct = inputFile_struct[f'{inputFile_struct.columns[1]}'].values
                                
                        # Define general information
                        imt         = currentIM.replace('s','')
                        uncert      = True
                        ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                        
                        loss_ratios = np.zeros((loss_ratios_struct.shape[0],1)) 
                        loss_ratios = loss_ratios_struct
    
                        # Perform the calculation to write to xml               
                        if uncert:
                            cov_loss=np.zeros(loss_ratios.shape)               
                            for m in range(loss_ratios.shape[0]):                        
                                mean_loss_ratio=loss_ratios[m]
                                if mean_loss_ratio<1e-4:
                                    loss_ratios[m]=1e-8
                                    cov_loss[m] = 1e-8
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss_ratios[m]= 0.99999
                                    cov_loss[m] = 1e-8
                                else:                                  
                                    sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                    cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                    
                        outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                        outFile.write('<imls imt="%s" > ' % imt)
                        outFile.writelines('%g ' % iml for iml in imls_struct)
                        outFile.write(' </imls>\n')
                        outFile.write('<meanLRs>')
                        outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                        outFile.write(' </meanLRs>\n')
                        outFile.write('<covLRs>')
                        outFile.writelines('%g ' % cv for cv in cov_loss)
                        outFile.write(' </covLRs>\n')
                        outFile.write('</vulnerabilityFunction>\n')
                                       
        # Close off the xml file 
        outFile.writelines('%s' % text for text in endText)    
        outFile.close()


def write_nonstructural_vulnerability_to_xml(building_classes, efficiency_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    IMT     = ['PGA','SA(0.3s)','SA(0.6s)','SA(1.0s)']         # Intensity measure labels
    IMT_XML = ['PGA','SA(0.3)','SA(0.6)','SA(1.0)']            # Intensity measure labels for xml outputs
    OCC     = ['RES','COM','IND']                                          # Occupancy Types

    # Define filename to print xml to 
    filename    = f'{vulnerability_out}/vulnerability_nonstructural.xml'
    
    startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
          '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
          '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="nonstructural">',
          '<description>nonstructural vulnerability model</description>']        
    endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
    
    outFile = open(filename, 'w')
    outFile.writelines("%s\n" % text for text in startText)
    
    # Loop over building classes and occupancy types  
    for i, currentBuildingClass in enumerate(building_classes):   
                      
        for k, currentIM in enumerate(IMT):
                                            
            if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                    
                for l, currentOccupancy in enumerate(OCC):
                         
                    # Load structural vulnerability input files
                    inputFile_nonstruct   = pd.read_csv(f'{vulnerability_in}/nonstructural/total/csv/nonstructural_total_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv')
                    
                    # Extract relevant information for structural vulnerability
                    imls_nonstruct        = inputFile_nonstruct[f'{inputFile_nonstruct.columns[0]}'].values                     
                    loss_ratios_nonstruct = inputFile_nonstruct[f'{inputFile_nonstruct.columns[1]}'].values
                            
                    # Define general information
                    imt         = IMT_XML[k]
                    uncert      = True
                    ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                                                            
                    loss_ratios = np.zeros((loss_ratios_nonstruct.shape[0],1)) 
                    loss_ratios = loss_ratios_nonstruct
    
                    # Perform the calculation to write to xml               
                    if uncert:
                        cov_loss=np.zeros(loss_ratios.shape)               
                        for m in range(loss_ratios.shape[0]):                        
                            mean_loss_ratio=loss_ratios[m]
                            if mean_loss_ratio<1e-4:
                                loss_ratios[m]=1e-8
                                cov_loss[m] = 1e-8
                            elif np.abs(1-mean_loss_ratio)<1e-4:
                                loss_ratios[m]= 0.99999
                                cov_loss[m] = 1e-8
                            else:                                  
                                sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                
                    outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                    outFile.write('<imls imt="%s" > ' % imt)
                    outFile.writelines('%g ' % iml for iml in imls_nonstruct)
                    outFile.write(' </imls>\n')
                    outFile.write('<meanLRs>')
                    outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                    outFile.write(' </meanLRs>\n')
                    outFile.write('<covLRs>')
                    outFile.writelines('%g ' % cv for cv in cov_loss)
                    outFile.write(' </covLRs>\n')
                    outFile.write('</vulnerabilityFunction>\n')
            
            else:
                
                pass
                       
    # Close off the xml file 
    outFile.writelines('%s' % text for text in endText)    
    outFile.close()

def write_nonstructural_vulnerability_to_xml_singleIM(building_classes, IMT, summary_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    OCC     = ['RES','COM','IND']                                          # Occupancy Types

    for k, currentIM in enumerate(IMT):
        
        naming_suffix = currentIM.replace('s','').replace('(','').replace(')','').replace('.','')
    
        # Define filename to print xml to 
        filename    = f'{vulnerability_out}/vulnerability_nonstructural_{naming_suffix}.xml'
        
        startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
              '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
              '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="nonstructural">',
              '<description>nonstructural vulnerability model</description>']        
        endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
        
        outFile = open(filename, 'w')
        outFile.writelines("%s\n" % text for text in startText)
        
        # Loop over building classes and occupancy types  
        for i, currentBuildingClass in enumerate(building_classes):   
                                                              
                    for l, currentOccupancy in enumerate(OCC):
                             
                        # Load structural vulnerability input files
                        inputFile_nonstruct   = pd.read_csv(f'{vulnerability_in}/nonstructural/total/csv/nonstructural_total_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv')
                        
                        # Extract relevant information for structural vulnerability
                        imls_nonstruct        = inputFile_nonstruct[f'{inputFile_nonstruct.columns[0]}'].values                     
                        loss_ratios_nonstruct = inputFile_nonstruct[f'{inputFile_nonstruct.columns[1]}'].values
                                
                        # Define general information
                        imt         = currentIM.replace('s','')
                        uncert      = True
                        ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                                                                
                        loss_ratios = np.zeros((loss_ratios_nonstruct.shape[0],1)) 
                        loss_ratios = loss_ratios_nonstruct
        
                        # Perform the calculation to write to xml               
                        if uncert:
                            cov_loss=np.zeros(loss_ratios.shape)               
                            for m in range(loss_ratios.shape[0]):                        
                                mean_loss_ratio=loss_ratios[m]
                                if mean_loss_ratio<1e-4:
                                    loss_ratios[m]=1e-8
                                    cov_loss[m] = 1e-8
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss_ratios[m]= 0.99999
                                    cov_loss[m] = 1e-8
                                else:                                  
                                    sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                    cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                    
                        outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                        outFile.write('<imls imt="%s" > ' % imt)
                        outFile.writelines('%g ' % iml for iml in imls_nonstruct)
                        outFile.write(' </imls>\n')
                        outFile.write('<meanLRs>')
                        outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                        outFile.write(' </meanLRs>\n')
                        outFile.write('<covLRs>')
                        outFile.writelines('%g ' % cv for cv in cov_loss)
                        outFile.write(' </covLRs>\n')
                        outFile.write('</vulnerabilityFunction>\n')
                                       
        # Close off the xml file 
        outFile.writelines('%s' % text for text in endText)    
        outFile.close()


def write_contents_vulnerability_to_xml(building_classes, efficiency_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    IMT     = ['PGA','SA(0.3s)','SA(0.6s)','SA(1.0s)']         # Intensity measure labels
    IMT_XML = ['PGA','SA(0.3)','SA(0.6)','SA(1.0)']            # Intensity measure labels for xml outputs
    OCC     = ['RES','COM','IND']                                          # Occupancy Types

    # Define filename to print xml to 
    filename    = f'{vulnerability_out}/vulnerability_contents.xml'
    
    startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
          '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
          '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="contents">',
          '<description>contents vulnerability model</description>']        
    endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
    
    outFile = open(filename, 'w')
    outFile.writelines("%s\n" % text for text in startText)
    
    # Loop over building classes and occupancy types  
    for i, currentBuildingClass in enumerate(building_classes):   
                      
        for k, currentIM in enumerate(IMT):
                                            
            if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                    
                for l, currentOccupancy in enumerate(OCC):
                                             
                    # Load structural vulnerability input files
                    inputFile_cont   = pd.read_csv(f'{vulnerability_in}/contents/csv/contents_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv')
                        
                    # Extract relevant information for structural vulnerability
                    imls_cont        = inputFile_cont[f'{inputFile_cont.columns[0]}'].values
                    loss_ratios_cont = inputFile_cont[f'{inputFile_cont.columns[1]}'].values
    
                    # Define general information
                    imt         = IMT_XML[k]
                    uncert      = True
                    ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                                                            
                    loss_ratios = np.zeros((loss_ratios_cont.shape[0],1)) 
                    loss_ratios = loss_ratios_cont
    
                    # Perform the calculation to write to xml               
                    if uncert:
                        cov_loss=np.zeros(loss_ratios.shape)               
                        for m in range(loss_ratios.shape[0]):                        
                            mean_loss_ratio=loss_ratios[m]
                            if mean_loss_ratio<1e-4:
                                loss_ratios[m]=1e-8
                                cov_loss[m] = 1e-8
                            elif np.abs(1-mean_loss_ratio)<1e-4:
                                loss_ratios[m]= 0.99999
                                cov_loss[m] = 1e-8
                            else:                                  
                                sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                
                    outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                    outFile.write('<imls imt="%s" > ' % imt)
                    outFile.writelines('%g ' % iml for iml in imls_cont)
                    outFile.write(' </imls>\n')
                    outFile.write('<meanLRs>')
                    outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                    outFile.write(' </meanLRs>\n')
                    outFile.write('<covLRs>')
                    outFile.writelines('%g ' % cv for cv in cov_loss)
                    outFile.write(' </covLRs>\n')
                    outFile.write('</vulnerabilityFunction>\n')
            
            else:
                
                pass
                       
    # Close off the xml file 
    outFile.writelines('%s' % text for text in endText)    
    outFile.close()
    
def write_contents_vulnerability_to_xml_singleIM(building_classes, IMT, summary_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    OCC     = ['RES','COM','IND']                                          # Occupancy Types

    for k, currentIM in enumerate(IMT):
        
        naming_suffix = currentIM.replace('s','').replace('(','').replace(')','').replace('.','')
    
        # Define filename to print xml to 
        filename    = f'{vulnerability_out}/vulnerability_contents_{naming_suffix}.xml'
        
        startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
              '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
              '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="contents">',
              '<description>contents vulnerability model</description>']        
        endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
        
        outFile = open(filename, 'w')
        outFile.writelines("%s\n" % text for text in startText)
        
        # Loop over building classes and occupancy types  
        for i, currentBuildingClass in enumerate(building_classes):   
                                                              
                    for l, currentOccupancy in enumerate(OCC):
                                                 
                        # Load structural vulnerability input files
                        inputFile_cont   = pd.read_csv(f'{vulnerability_in}/contents/csv/contents_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv')
                            
                        # Extract relevant information for structural vulnerability
                        imls_cont        = inputFile_cont[f'{inputFile_cont.columns[0]}'].values
                        loss_ratios_cont = inputFile_cont[f'{inputFile_cont.columns[1]}'].values
        
                        # Define general information
                        imt         = currentIM.replace('s','')
                        uncert      = True
                        ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                                                                
                        loss_ratios = np.zeros((loss_ratios_cont.shape[0],1)) 
                        loss_ratios = loss_ratios_cont
        
                        # Perform the calculation to write to xml               
                        if uncert:
                            cov_loss=np.zeros(loss_ratios.shape)               
                            for m in range(loss_ratios.shape[0]):                        
                                mean_loss_ratio=loss_ratios[m]
                                if mean_loss_ratio<1e-4:
                                    loss_ratios[m]=1e-8
                                    cov_loss[m] = 1e-8
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss_ratios[m]= 0.99999
                                    cov_loss[m] = 1e-8
                                else:                                  
                                    sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                    cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                    
                        outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                        outFile.write('<imls imt="%s" > ' % imt)
                        outFile.writelines('%g ' % iml for iml in imls_cont)
                        outFile.write(' </imls>\n')
                        outFile.write('<meanLRs>')
                        outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                        outFile.write(' </meanLRs>\n')
                        outFile.write('<covLRs>')
                        outFile.writelines('%g ' % cv for cv in cov_loss)
                        outFile.write(' </covLRs>\n')
                        outFile.write('</vulnerabilityFunction>\n')
                                       
        # Close off the xml file 
        outFile.writelines('%s' % text for text in endText)    
        outFile.close()
    
def write_fatality_vulnerability_to_xml(building_classes, efficiency_df, consequence_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    IMT     = ['PGA','SA(0.3s)','SA(0.6s)','SA(1.0s)']         # Intensity measure labels
    IMT_XML = ['PGA','SA(0.3)','SA(0.6)','SA(1.0)']            # Intensity measure labels for xml outputs
    OCC     = ['RES','COM','IND']                                          # Occupancy Types
    INTENSITIES = np.round(np.geomspace(0.05, 10.0, 50), 3)
    
    # Define filename to print xml to 
    filename    = f'{vulnerability_out}/vulnerability_fatalities.xml'
    
    startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
          '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
          '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="occupants">',
          '<description> Vulnerability model to calculate fatalities</description>']        
    endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
    
    outFile = open(filename, 'w')
    outFile.writelines("%s\n" % text for text in startText)
    
    # Loop over building classes and occupancy types  
    for i, currentBuildingClass in enumerate(building_classes):   
                 
        for k, currentIM in enumerate(IMT):
                                        
            if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                    
                for l, currentOccupancy in enumerate(OCC):

                    # Define general information
                    imt         = IMT_XML[k]
                    uncert      = True
                    ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                       
                    # Load structural DS4 input files
                    thetas = efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 3:7].values.tolist()
                    betas  = [efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 7]]*4                              
                    
                    # Load the consequence model
                    currentConsequence = consequence_df.iloc[consequence_df[consequence_df.taxonomy == ID].index[0], 3:7].values.tolist()
                    
                    imls = INTENSITIES
                    array = calculate_structural_vulnerability(thetas, betas, currentConsequence)
                    loss_ratios = array[:,1]
                    
                    # Perform the calculation to write to xml               
                    if uncert:
                        cov_loss=np.zeros(loss_ratios.shape)               
                        for m in range(loss_ratios.shape[0]):                        
                            mean_loss_ratio=loss_ratios[m]
                            if mean_loss_ratio<1e-4:
                                loss_ratios[m]=1e-8
                                cov_loss[m] = 1e-8
                            elif np.abs(1-mean_loss_ratio)<1e-4:
                                loss_ratios[m]= 0.99999
                                cov_loss[m] = 1e-8
                            else:                                  
                                sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                
                    outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                    outFile.write('<imls imt="%s" > ' % imt)
                    outFile.writelines('%g ' % iml for iml in imls)
                    outFile.write(' </imls>\n')
                    outFile.write('<meanLRs>')
                    outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                    outFile.write(' </meanLRs>\n')
                    outFile.write('<covLRs>')
                    outFile.writelines('%g ' % cv for cv in cov_loss)
                    outFile.write(' </covLRs>\n')
                    outFile.write('</vulnerabilityFunction>\n')
            
            else:
                
                pass
                       
    # Close off the xml file 
    outFile.writelines('%s' % text for text in endText)    
    outFile.close()
    
def write_fatality_vulnerability_to_xml_singleIM(building_classes, IMT, summary_df, consequence_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    OCC     = ['RES','COM','IND']                                          # Occupancy Types
    INTENSITIES = np.round(np.geomspace(0.05, 10.0, 50), 3)

    for k, currentIM in enumerate(IMT):
        
        naming_suffix = currentIM.replace('s','').replace('(','').replace(')','').replace('.','')
    
        
        # Define filename to print xml to 
        filename    = f'{vulnerability_out}/vulnerability_fatalities_{naming_suffix}.xml'
        
        startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
              '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
              '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="occupants">',
              '<description> Vulnerability model to calculate fatalities</description>']        
        endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
        
        outFile = open(filename, 'w')
        outFile.writelines("%s\n" % text for text in startText)
        
        # Loop over building classes and occupancy types  
        for i, currentBuildingClass in enumerate(building_classes):   
                                                         
                    for l, currentOccupancy in enumerate(OCC):
    
                        # Define general information
                        imt         = currentIM.replace('s','')
                        uncert      = True
                        ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                           
                        # Load structural DS4 input files
                        thetas = summary_df.iloc[summary_df[summary_df.Building_class == currentBuildingClass].index[0], 1:5].values.tolist()
                        betas  = [summary_df.iloc[summary_df[summary_df.Building_class == currentBuildingClass].index[0], 5]]*4                              
                        
                        # Load the consequence model
                        currentConsequence = consequence_df.iloc[consequence_df[consequence_df.taxonomy == ID].index[0], 3:7].values.tolist()
                        
                        imls = INTENSITIES
                        array = calculate_structural_vulnerability(thetas, betas, currentConsequence)
                        loss_ratios = array[:,1]
                        
                        # Perform the calculation to write to xml               
                        if uncert:
                            cov_loss=np.zeros(loss_ratios.shape)               
                            for m in range(loss_ratios.shape[0]):                        
                                mean_loss_ratio=loss_ratios[m]
                                if mean_loss_ratio<1e-4:
                                    loss_ratios[m]=1e-8
                                    cov_loss[m] = 1e-8
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss_ratios[m]= 0.99999
                                    cov_loss[m] = 1e-8
                                else:                                  
                                    sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                    cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                    
                        outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                        outFile.write('<imls imt="%s" > ' % imt)
                        outFile.writelines('%g ' % iml for iml in imls)
                        outFile.write(' </imls>\n')
                        outFile.write('<meanLRs>')
                        outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                        outFile.write(' </meanLRs>\n')
                        outFile.write('<covLRs>')
                        outFile.writelines('%g ' % cv for cv in cov_loss)
                        outFile.write(' </covLRs>\n')
                        outFile.write('</vulnerabilityFunction>\n')
                                       
        # Close off the xml file 
        outFile.writelines('%s' % text for text in endText)    
        outFile.close()
    
def write_injury_vulnerability_to_xml(building_classes, efficiency_df, consequence_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    IMT     = ['PGA','SA(0.3s)','SA(0.6s)','SA(1.0s)']         # Intensity measure labels
    IMT_XML = ['PGA','SA(0.3)','SA(0.6)','SA(1.0)']            # Intensity measure labels for xml outputs
    OCC     = ['RES','COM','IND']                                          # Occupancy Types
    INTENSITIES = np.round(np.geomspace(0.05, 10.0, 50), 3)
    
    # Define filename to print xml to 
    filename    = f'{vulnerability_out}/vulnerability_injuries.xml'
    
    startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
          '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
          '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="occupants">',
          '<description> Vulnerability model to calculate injuries</description>']        
    endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
    
    outFile = open(filename, 'w')
    outFile.writelines("%s\n" % text for text in startText)
    
    # Loop over building classes and occupancy types  
    for i, currentBuildingClass in enumerate(building_classes):   
                 
        for k, currentIM in enumerate(IMT):
                                        
            if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                    
                for l, currentOccupancy in enumerate(OCC):

                    # Define general information
                    imt         = IMT_XML[k]
                    uncert      = True
                    ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                       
                    # Load structural DS4 input files
                    thetas = efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 3:7].values.tolist()
                    betas  = [efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 7]]*4                              
                    
                    # Load the consequence model
                    currentConsequence = consequence_df.iloc[consequence_df[consequence_df.taxonomy == ID].index[0], 3:7].values.tolist()
                    
                    imls = INTENSITIES
                    array = calculate_structural_vulnerability(thetas, betas, currentConsequence)
                    loss_ratios = array[:,1]
                    
                    # Perform the calculation to write to xml               
                    if uncert:
                        cov_loss=np.zeros(loss_ratios.shape)               
                        for m in range(loss_ratios.shape[0]):                        
                            mean_loss_ratio=loss_ratios[m]
                            if mean_loss_ratio<1e-4:
                                loss_ratios[m]=1e-8
                                cov_loss[m] = 1e-8
                            elif np.abs(1-mean_loss_ratio)<1e-4:
                                loss_ratios[m]= 0.99999
                                cov_loss[m] = 1e-8
                            else:                                  
                                sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                
                    outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                    outFile.write('<imls imt="%s" > ' % imt)
                    outFile.writelines('%g ' % iml for iml in imls)
                    outFile.write(' </imls>\n')
                    outFile.write('<meanLRs>')
                    outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                    outFile.write(' </meanLRs>\n')
                    outFile.write('<covLRs>')
                    outFile.writelines('%g ' % cv for cv in cov_loss)
                    outFile.write(' </covLRs>\n')
                    outFile.write('</vulnerabilityFunction>\n')
            
            else:
                
                pass
                       
    # Close off the xml file 
    outFile.writelines('%s' % text for text in endText)    
    outFile.close()
    
def write_injury_vulnerability_to_xml_singleIM(building_classes, IMT, summary_df, consequence_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    OCC     = ['RES','COM','IND']                                          # Occupancy Types
    INTENSITIES = np.round(np.geomspace(0.05, 10.0, 50), 3)

    for k, currentIM in enumerate(IMT):
        
        naming_suffix = currentIM.replace('s','').replace('(','').replace(')','').replace('.','')
        
        # Define filename to print xml to 
        filename    = f'{vulnerability_out}/vulnerability_injuries_{naming_suffix}.xml'
        
        startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
              '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
              '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="occupants">',
              '<description> Vulnerability model to calculate injuries</description>']        
        endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
        
        outFile = open(filename, 'w')
        outFile.writelines("%s\n" % text for text in startText)
        
        # Loop over building classes and occupancy types  
        for i, currentBuildingClass in enumerate(building_classes):   
                                                         
                    for l, currentOccupancy in enumerate(OCC):
    
                        # Define general information
                        imt         = currentIM.replace('s','')
                        uncert      = True
                        ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                           
                        # Load structural DS4 input files
                        thetas = summary_df.iloc[summary_df[summary_df.Building_class == currentBuildingClass].index[0], 1:5].values.tolist()
                        betas  = [summary_df.iloc[summary_df[summary_df.Building_class == currentBuildingClass].index[0], 5]]*4                              
                        
                        # Load the consequence model
                        currentConsequence = consequence_df.iloc[consequence_df[consequence_df.taxonomy == ID].index[0], 3:7].values.tolist()
                        
                        imls = INTENSITIES
                        array = calculate_structural_vulnerability(thetas, betas, currentConsequence)
                        loss_ratios = array[:,1]
                        
                        # Perform the calculation to write to xml               
                        if uncert:
                            cov_loss=np.zeros(loss_ratios.shape)               
                            for m in range(loss_ratios.shape[0]):                        
                                mean_loss_ratio=loss_ratios[m]
                                if mean_loss_ratio<1e-4:
                                    loss_ratios[m]=1e-8
                                    cov_loss[m] = 1e-8
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss_ratios[m]= 0.99999
                                    cov_loss[m] = 1e-8
                                else:                                  
                                    sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                    cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                    
                        outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                        outFile.write('<imls imt="%s" > ' % imt)
                        outFile.writelines('%g ' % iml for iml in imls)
                        outFile.write(' </imls>\n')
                        outFile.write('<meanLRs>')
                        outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                        outFile.write(' </meanLRs>\n')
                        outFile.write('<covLRs>')
                        outFile.writelines('%g ' % cv for cv in cov_loss)
                        outFile.write(' </covLRs>\n')
                        outFile.write('</vulnerabilityFunction>\n')
                                       
        # Close off the xml file 
        outFile.writelines('%s' % text for text in endText)    
        outFile.close()

def write_homeless_vulnerability_to_xml(building_classes, efficiency_df, consequence_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    IMT     = ['PGA','SA(0.3s)','SA(0.6s)','SA(1.0s)']         # Intensity measure labels
    IMT_XML = ['PGA','SA(0.3)','SA(0.6)','SA(1.0)']            # Intensity measure labels for xml outputs
    OCC     = ['RES','COM','IND']                                          # Occupancy Types
    INTENSITIES = np.round(np.geomspace(0.05, 10.0, 50), 3)
    
    # Define filename to print xml to 
    filename    = f'{vulnerability_out}/vulnerability_residents.xml'
    
    startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
          '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
          '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="residents">',
          '<description> Vulnerability model to calculate displaced population (homeless)</description>']        
    endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
    
    outFile = open(filename, 'w')
    outFile.writelines("%s\n" % text for text in startText)
    
    # Loop over building classes and occupancy types  
    for i, currentBuildingClass in enumerate(building_classes):   
                 
        for k, currentIM in enumerate(IMT):
                                        
            if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                    
                for l, currentOccupancy in enumerate(OCC):

                    # Define general information
                    imt         = IMT_XML[k]
                    uncert      = True
                    ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                       
                    # Load structural DS4 input files
                    thetas = efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 3:7].values.tolist()
                    betas  = [efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 7]]*4                              
                    
                    # Load the consequence model
                    currentConsequence = consequence_df.iloc[consequence_df[consequence_df.taxonomy == ID].index[0], 3:7].values.tolist()
                    
                    imls = INTENSITIES
                    array = calculate_structural_vulnerability(thetas, betas, currentConsequence)
                    loss_ratios = array[:,1]
                    
                    # Perform the calculation to write to xml               
                    if uncert:
                        cov_loss=np.zeros(loss_ratios.shape)               
                        for m in range(loss_ratios.shape[0]):                        
                            mean_loss_ratio=loss_ratios[m]
                            if mean_loss_ratio<1e-4:
                                loss_ratios[m]=1e-8
                                cov_loss[m] = 1e-8
                            elif np.abs(1-mean_loss_ratio)<1e-4:
                                loss_ratios[m]= 0.99999
                                cov_loss[m] = 1e-8
                            else:                                  
                                sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                
                    outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                    outFile.write('<imls imt="%s" > ' % imt)
                    outFile.writelines('%g ' % iml for iml in imls)
                    outFile.write(' </imls>\n')
                    outFile.write('<meanLRs>')
                    outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                    outFile.write(' </meanLRs>\n')
                    outFile.write('<covLRs>')
                    outFile.writelines('%g ' % cv for cv in cov_loss)
                    outFile.write(' </covLRs>\n')
                    outFile.write('</vulnerabilityFunction>\n')
            
            else:
                
                pass
                       
    # Close off the xml file 
    outFile.writelines('%s' % text for text in endText)    
    outFile.close()

def write_homeless_vulnerability_to_xml_singleIM(building_classes, IMT, summary_df, consequence_df, vulnerability_in, vulnerability_out):
    
    # Define constants
    OCC     = ['RES','COM','IND']                                          # Occupancy Types
    INTENSITIES = np.round(np.geomspace(0.05, 10.0, 50), 3)

    for k, currentIM in enumerate(IMT):
        
        naming_suffix = currentIM.replace('s','').replace('(','').replace(')','').replace('.','')
        
        # Define filename to print xml to 
        filename    = f'{vulnerability_out}/vulnerability_residents_{naming_suffix}.xml'
        
        startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
              '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
              '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="residents">',
              '<description> Vulnerability model to calculate displaced population (homeless)</description>']        
        endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
        
        outFile = open(filename, 'w')
        outFile.writelines("%s\n" % text for text in startText)
        
        # Loop over building classes and occupancy types  
        for i, currentBuildingClass in enumerate(building_classes):   
                                                         
                    for l, currentOccupancy in enumerate(OCC):
    
                        # Define general information
                        imt         = currentIM.replace('s','')
                        uncert      = True
                        ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                           
                        # Load structural DS4 input files
                        thetas = summary_df.iloc[summary_df[summary_df.Building_class == currentBuildingClass].index[0], 1:5].values.tolist()
                        betas  = [summary_df.iloc[summary_df[summary_df.Building_class == currentBuildingClass].index[0], 5]]*4                              
                        
                        # Load the consequence model
                        currentConsequence = consequence_df.iloc[consequence_df[consequence_df.taxonomy == ID].index[0], 3:7].values.tolist()
                        
                        imls = INTENSITIES
                        array = calculate_structural_vulnerability(thetas, betas, currentConsequence)
                        loss_ratios = array[:,1]
                        
                        # Perform the calculation to write to xml               
                        if uncert:
                            cov_loss=np.zeros(loss_ratios.shape)               
                            for m in range(loss_ratios.shape[0]):                        
                                mean_loss_ratio=loss_ratios[m]
                                if mean_loss_ratio<1e-4:
                                    loss_ratios[m]=1e-8
                                    cov_loss[m] = 1e-8
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss_ratios[m]= 0.99999
                                    cov_loss[m] = 1e-8
                                else:                                  
                                    sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                    cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                    
                        outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                        outFile.write('<imls imt="%s" > ' % imt)
                        outFile.writelines('%g ' % iml for iml in imls)
                        outFile.write(' </imls>\n')
                        outFile.write('<meanLRs>')
                        outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                        outFile.write(' </meanLRs>\n')
                        outFile.write('<covLRs>')
                        outFile.writelines('%g ' % cv for cv in cov_loss)
                        outFile.write(' </covLRs>\n')
                        outFile.write('</vulnerabilityFunction>\n')
                                       
        # Close off the xml file 
        outFile.writelines('%s' % text for text in endText)    
        outFile.close()

def write_structural_fragility_to_xml(building_classes, efficiency_df, vulnerability_in, vulnerability_out):

    # Define constants
    IMT     = ['PGA','SA(0.3s)','SA(0.6s)','SA(1.0s)']         # Intensity measure labels
    IMT_XML = ['PGA','SA(0.3)','SA(0.6)','SA(1.0)']            # Intensity measure labels for xml outputs
    OCC     = ['RES','COM','IND']                                          # Occupancy Types
    INTENSITIES = np.round(np.geomspace(0.05, 10.0, 50), 3)


    # Define filename to print xml to 
    filename    = f'{vulnerability_out}/fragility_structural.xml'
    
    startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
          '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
          '<fragilityModel id="fragility_model" assetCategory="buildings" lossCategory="structural">',
          '<description> Structural fragility model </description>',
          '<limitStates>slight moderate extensive complete </limitStates>']        
    endText = ['\n</fragilityModel>\n', '</nrml>\n']
    
    outFile = open(filename, 'w')
    outFile.writelines("%s\n" % text for text in startText)

        
    # Loop over building classes and occupancy types  
    for i, currentBuildingClass in enumerate(building_classes):   
                 
        for k, currentIM in enumerate(IMT):
                                        
            if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                    
                for l, currentOccupancy in enumerate(OCC):

                    # Define general information
                    imt         = IMT_XML[k]
                    uncert      = True
                    ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                       
                    # Load nonstructural fragility parameters
                    thetas = efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 3:7].values.tolist()
                    betas  = [efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 7]]*4                              
                    
                    # Create fragility functions
                    p_ds1 = get_fragility_function(INTENSITIES, thetas[0], betas[0])   
                    p_ds2 = get_fragility_function(INTENSITIES, thetas[1], betas[1])   
                    p_ds3 = get_fragility_function(INTENSITIES, thetas[2], betas[2]) 
                    p_ds4 = get_fragility_function(INTENSITIES, thetas[3], betas[3]) 

                    outFile.write('\n<fragilityFunction id="%s" format="Discrete">\n' % ID)      
                    outFile.write('<imls imt="%s" noDamageLimit="0.05"> ' % imt)
                    outFile.writelines('%g ' % iml for iml in INTENSITIES)
                    outFile.write(' </imls>\n')
                    
                    outFile.write('<poes ls="slight">')
                    outFile.writelines('%g ' % poes for poes in p_ds1)
                    outFile.write(' </poes>\n')
                    
                    outFile.write('<poes ls="moderate">')
                    outFile.writelines('%g ' % poes for poes in p_ds2)
                    outFile.write(' </poes>\n')
    
                    outFile.write('<poes ls="extensive">')
                    outFile.writelines('%g ' % poes for poes in p_ds3)
                    outFile.write(' </poes>\n')
    
                    outFile.write('<poes ls="complete">')
                    outFile.writelines('%g ' % poes for poes in p_ds4)
                    outFile.write(' </poes>\n')
                    outFile.write('</fragilityFunction>\n')
                                
            else:
                
                pass
                       
    # Close off the xml file 
    outFile.writelines('%s' % text for text in endText)    
    outFile.close()


def write_nonstructural_fragility_to_xml(building_classes, efficiency_df, vulnerability_in, vulnerability_out):

    # Define constants
    IMT     = ['PGA','SA(0.3s)','SA(0.6s)','SA(1.0s)']         # Intensity measure labels
    IMT_XML = ['PGA','SA(0.3)','SA(0.6)','SA(1.0)']            # Intensity measure labels for xml outputs
    OCC     = ['RES','COM','IND']                                          # Occupancy Types
    INTENSITIES = np.round(np.geomspace(0.05, 10.0, 50), 3)


    # Define filename to print xml to 
    filename    = f'{vulnerability_out}/fragility_nonstructural_drift.xml'
    
    startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
          '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
          '<fragilityModel id="fragility_model" assetCategory="buildings" lossCategory="nonstructural">',
          '<description> Structural fragility model </description>',
          '<limitStates>slight moderate extensive complete </limitStates>']        
    endText = ['\n</fragilityModel>\n', '</nrml>\n']
    
    outFile = open(filename, 'w')
    outFile.writelines("%s\n" % text for text in startText)

    # Loop over building classes and occupancy types  
    for i, currentBuildingClass in enumerate(building_classes):   
                 
        for k, currentIM in enumerate(IMT):
                                        
            if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                    
                for l, currentOccupancy in enumerate(OCC):
                    
                    # Define general information
                    imt         = IMT_XML[k]
                    uncert      = True
                    ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                       
                    # Load structural fragility parameters
                    fragility_df = pd.read_csv(f'{vulnerability_in}/nonstructural/drift/csv/nonstructural_drift_fragility_{currentIM}_summary.csv')
                    thetas = fragility_df.iloc[fragility_df[fragility_df.Building_class == currentBuildingClass].index[0], 1:5].values.tolist()
                    betas  = [fragility_df.iloc[fragility_df[fragility_df.Building_class == currentBuildingClass].index[0], 5]]*4                              
                    
                    # Create fragility functions
                    p_ds1 = get_fragility_function(INTENSITIES, thetas[0], betas[0])   
                    p_ds2 = get_fragility_function(INTENSITIES, thetas[1], betas[1])   
                    p_ds3 = get_fragility_function(INTENSITIES, thetas[2], betas[2]) 
                    p_ds4 = get_fragility_function(INTENSITIES, thetas[3], betas[3]) 

                    outFile.write('\n<fragilityFunction id="%s" format="Discrete">\n' % ID)      
                    outFile.write('<imls imt="%s" noDamageLimit="0.05"> ' % imt)
                    outFile.writelines('%g ' % iml for iml in INTENSITIES)
                    outFile.write(' </imls>\n')
                    
                    outFile.write('<poes ls="slight">')
                    outFile.writelines('%g ' % poes for poes in p_ds1)
                    outFile.write(' </poes>\n')
                    
                    outFile.write('<poes ls="moderate">')
                    outFile.writelines('%g ' % poes for poes in p_ds2)
                    outFile.write(' </poes>\n')
    
                    outFile.write('<poes ls="extensive">')
                    outFile.writelines('%g ' % poes for poes in p_ds3)
                    outFile.write(' </poes>\n')
    
                    outFile.write('<poes ls="complete">')
                    outFile.writelines('%g ' % poes for poes in p_ds4)
                    outFile.write(' </poes>\n')
                    outFile.write('</fragilityFunction>\n')
                                
            else:
                
                pass
                       
    # Close off the xml file 
    outFile.writelines('%s' % text for text in endText)    
    outFile.close()
    
def write_contents_fragility_to_xml(building_classes, efficiency_df, vulnerability_in, vulnerability_out):

    # Define constants
    IMT     = ['PGA','SA(0.3s)','SA(0.6s)','SA(1.0s)']         # Intensity measure labels
    IMT_XML = ['PGA','SA(0.3)','SA(0.6)','SA(1.0)']            # Intensity measure labels for xml outputs
    OCC     = ['RES','COM','IND']                                          # Occupancy Types
    INTENSITIES = np.round(np.geomspace(0.05, 10.0, 50), 3)


    # Define filename to print xml to 
    filename    = f'{vulnerability_out}/fragility_contents.xml'
    
    startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
          '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
          '<fragilityModel id="fragility_model" assetCategory="buildings" lossCategory="contents">',
          '<description> Structural fragility model </description>',
          '<limitStates>slight moderate extensive complete </limitStates>']        
    endText = ['\n</fragilityModel>\n', '</nrml>\n']
    
    outFile = open(filename, 'w')
    outFile.writelines("%s\n" % text for text in startText)

    # Loop over building classes and occupancy types  
    for i, currentBuildingClass in enumerate(building_classes):   
                 
        for k, currentIM in enumerate(IMT):
                                        
            if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                    
                for l, currentOccupancy in enumerate(OCC):
                    
                    # Define general information
                    imt         = IMT_XML[k]
                    uncert      = True
                    ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                       
                    # Load contents fragility file 
                    inputFile   = pd.read_csv(f'{vulnerability_in}/contents/csv/contents_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv', header=None).values
                    
                    # Create fragility functions
                    p_ds4 = inputFile[:,1]

                    outFile.write('\n<fragilityFunction id="%s" format="Discrete">\n' % ID)      
                    outFile.write('<imls imt="%s" noDamageLimit="0.05"> ' % imt)
                    outFile.writelines('%g ' % iml for iml in INTENSITIES)
                    outFile.write(' </imls>\n')
                        
                    outFile.write('<poes ls="complete">')
                    outFile.writelines('%g ' % poes for poes in p_ds4)
                    outFile.write(' </poes>\n')
                    outFile.write('</fragilityFunction>\n')
                                
            else:
                
                pass
                       
    # Close off the xml file 
    outFile.writelines('%s' % text for text in endText)    
    outFile.close()
    
    
def write_vulnerability_to_xml_2(efficiency_df, consequenceType, vulnerability_in, vulnerability_out, consequence_df = []):
    """
    Exports the vulnerability functions associated with the efficient intensity
    measures to XML files readable by the OpenQuake Engine
    
    Parameters
    ----------
    building_classes:              list                List of strings corresponding to the taxonomy codes representing 
                                                       the covered building classes 
    efficiency_df:                array                Efficiency summary DataFrame exported following nonlinear time-history analyses post-processing 
    consequenceType:             string                Consequence type (options are:
                                                                         'DAM': loss due to damage (if 'DAM' is provided then it will export the structural, non-structural and contents
                                                                                        vulnerabilities to XML at once in seperate files),
                                                                         'AREA': damaged area,
                                                                         'NUMBER': number of damaged buildings,
                                                                         'INJURIES': severe injuries rate,
                                                                         'FATALITIES': fatalities rate,
                                                                         'DISPLACED': displaced population rate,
                                                                         for the latter three consequence typologies, a consequence dataframe equal in size to the building classes
                                                                         must be provided)
                                
    vulnerability_in:   directory string               The file path to the vulnerability functions
    vulnerability_out:  directory string               The file path where to export the XML files

    Returns
    -------
    None
    """        

    # Define constants
    IMT     = ['PGA','SA(0.3s)','SA(0.6s)','SA(1.0s)']         # Intensity measure labels
    IMT_XML = ['PGA','SA(0.3)','SA(0.6)','SA(1.0)']            # Intensity measure labels for xml outputs
    OCC     = ['RES','COM','IND']                              # Occupancy Types
    COMPONENTS = ['structural','nonstructural','contents']     # Component Types
    INTENSITIES = np.round(np.geomspace(0.05, 10.0, 50), 3)    # Intensity measure levels
    
    ## Get the building classes in the efficiency DataFrame
    building_classes = efficiency_df.Building_class.tolist()
    
    if consequenceType == 'DAM':
        
        for i, currentComponent in enumerate(COMPONENTS):   
             
            # Define filename to print xml to 
            filename    = f'{vulnerability_out}/vulnerability_{currentComponent}.xml'
            
            if currentComponent == 'structural':        
                startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
                      '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
                      '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="structural">',
                      '<description>structural vulnerability model</description>']        
                endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
    
            elif currentComponent == 'nonstructural':
                startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
                      '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
                      '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="nonstructural">',
                      '<description>nonstructural vulnerability model</description>']
                endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
            
            elif currentComponent == 'contents':
                startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
                      '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
                      '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="contents">',
                      '<description>contents vulnerability model</description>']
                endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
            
            outFile = open(filename, 'w')
            outFile.writelines("%s\n" % text for text in startText)
    
            # Loop over building classes, intensity measures and occupancy types (in that order!)
            for j, currentBuildingClass in enumerate(building_classes):
                 
                for k, currentIM in enumerate(IMT):
                    
                    if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                        
                        for l, currentOccupancy in enumerate(OCC):
                            
                                               
                            # Extract information
                            if currentComponent =='structural':
                                loss_ratios   = pd.read_csv(f'{vulnerability_in}/structural/csv/structural_vulnerability_{currentBuildingClass}_{currentIM}.csv', header=None)[1].values
                            elif currentComponent == 'contents':
                                loss_ratios   = pd.read_csv(f'{vulnerability_in}/contents/csv/contents_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv', header=None)[1].values
                            elif currentComponent == 'nonstructural':
                                loss_ratios   = pd.read_csv(f'{vulnerability_in}/nonstructural/total/csv/nonstructural_total_vulnerability_{currentBuildingClass}_{currentIM}_{currentOccupancy}.csv', header=None)[1].values
                            
                            ID = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                            imt         = IMT_XML[k]
                            
                            # Perform the calculation to write to xml               
                            cov_loss=np.zeros(loss_ratios.shape)               
    
                            for m in range(loss_ratios.shape[0]):
                                
                                mean_loss_ratio=loss_ratios[m]
                                
                                if mean_loss_ratio<1e-4:
                                    loss_ratios[m]=1e-8
                                    cov_loss[m] = 1e-8
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss_ratios[m]= 0.99999
                                    cov_loss[m] = 1e-8
                                else:                                  
                                    sigma_loss, _, _ = calculate_sigma_loss(mean_loss_ratio)
                                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                    cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                                                      
                            outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                            outFile.write('<imls imt="%s" > ' % imt)
                            outFile.writelines('%g ' % iml for iml in INTENSITIES)
                            outFile.write(' </imls>\n')
                            outFile.write('<meanLRs>')
                            outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                            outFile.write(' </meanLRs>\n')
                            outFile.write('<covLRs>')
                            outFile.writelines('%g ' % cv for cv in cov_loss)
                            outFile.write(' </covLRs>\n')
                            outFile.write('</vulnerabilityFunction>\n')
                    
                    else:
                        
                        pass
            
            # Close off the xml file 
            outFile.writelines('%s' % text for text in endText)    
            outFile.close()

    elif consequenceType == 'DISPLACED':
        
        if consequence_df == []:
            raise ValueError('Consequence model needs to be included as input argument!')
        else:
            
            # Define filename to print xml to 
            filename    = f'{vulnerability_out}/vulnerability_residents.xml'
            
            startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
                  '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
                  '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="residents">',
                  '<description> Vulnerability model to calculate displaced population (homeless)</description>']        
            endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
            
            outFile = open(filename, 'w')
            outFile.writelines("%s\n" % text for text in startText)
            
            # Loop over building classes and occupancy types  
            for i, currentBuildingClass in enumerate(building_classes):   
                         
                for k, currentIM in enumerate(IMT):
                                                
                    if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                            
                        for l, currentOccupancy in enumerate(OCC):
                            

                            # Define general information
                            imt         = IMT_XML[k]
                            ID = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                               
                            # Load structural DS4 input files
                            thetas = efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 3:7].values.tolist()
                            betas  = [efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 7]]*4                              
                            
                            # Load the consequence model
                            currentConsequence = consequence_df.iloc[consequence_df[consequence_df.taxonomy == ID].index[0], 3:7].values.tolist()
                            
                            array = calculate_structural_vulnerability(thetas, betas, currentConsequence)
                            loss_ratios = array[:,1]
                            
                            # Perform the calculation to write to xml               
                            cov_loss=np.zeros(loss_ratios.shape)               
                            for m in range(loss_ratios.shape[0]):                        
                                mean_loss_ratio=loss_ratios[m]
                                if mean_loss_ratio<1e-4:
                                    loss_ratios[m]=1e-8
                                    cov_loss[m] = 1e-8
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss_ratios[m]= 0.99999
                                    cov_loss[m] = 1e-8
                                else:                                  
                                    sigma_loss, _, _ = calculate_sigma_loss(mean_loss_ratio)
                                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                    cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                        
                            outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                            outFile.write('<imls imt="%s" > ' % imt)
                            outFile.writelines('%g ' % iml for iml in INTENSITIES)
                            outFile.write(' </imls>\n')
                            outFile.write('<meanLRs>')
                            outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                            outFile.write(' </meanLRs>\n')
                            outFile.write('<covLRs>')
                            outFile.writelines('%g ' % cv for cv in cov_loss)
                            outFile.write(' </covLRs>\n')
                            outFile.write('</vulnerabilityFunction>\n')
                    
                    else:
                        
                        pass
                               
            # Close off the xml file 
            outFile.writelines('%s' % text for text in endText)    
            outFile.close()
            
    elif consequenceType == 'INJURIES':
        
        if consequence_df == []:
            raise ValueError('Consequence model needs to be included as input argument!')
        else:
            
            # Define filename to print xml to 
            filename    = f'{vulnerability_out}/vulnerability_injuries.xml'
            
            startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
                  '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
                  '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="occupants">',
                  '<description> Vulnerability model to calculate injuries</description>']        
            endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
            
            outFile = open(filename, 'w')
            outFile.writelines("%s\n" % text for text in startText)
            
            # Loop over building classes and occupancy types  
            for i, currentBuildingClass in enumerate(building_classes):   
                         
                for k, currentIM in enumerate(IMT):
                                                
                    if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                            
                        for l, currentOccupancy in enumerate(OCC):
    
                            # Define general information
                            imt         = IMT_XML[k]
                            ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                               
                            # Load structural DS4 input files
                            thetas = efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 3:7].values.tolist()
                            betas  = [efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 7]]*4                              
                            
                            # Load the consequence model
                            currentConsequence = consequence_df.iloc[consequence_df[consequence_df.taxonomy == ID].index[0], 3:7].values.tolist()
                            
                            array = calculate_structural_vulnerability(thetas, betas, currentConsequence)
                            loss_ratios = array[:,1]
                            
                            # Perform the calculation to write to xml               
                            cov_loss=np.zeros(loss_ratios.shape)               
                            for m in range(loss_ratios.shape[0]):                        
                                mean_loss_ratio=loss_ratios[m]
                                if mean_loss_ratio<1e-4:
                                    loss_ratios[m]=1e-8
                                    cov_loss[m] = 1e-8
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss_ratios[m]= 0.99999
                                    cov_loss[m] = 1e-8
                                else:                                  
                                    sigma_loss, _, _ = calculate_sigma_loss(mean_loss_ratio)
                                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                    cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                        
                            outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                            outFile.write('<imls imt="%s" > ' % imt)
                            outFile.writelines('%g ' % iml for iml in INTENSITIES)
                            outFile.write(' </imls>\n')
                            outFile.write('<meanLRs>')
                            outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                            outFile.write(' </meanLRs>\n')
                            outFile.write('<covLRs>')
                            outFile.writelines('%g ' % cv for cv in cov_loss)
                            outFile.write(' </covLRs>\n')
                            outFile.write('</vulnerabilityFunction>\n')
                    
                    else:
                        
                        pass
                               
            # Close off the xml file 
            outFile.writelines('%s' % text for text in endText)    
            outFile.close()
            
    elif consequenceType == 'FATALITIES':
        
        if consequence_df == []:
            raise ValueError('Consequence model needs to be included as input argument!')
        else:
            
            # Define filename to print xml to 
            filename    = f'{vulnerability_out}/vulnerability_fatalities.xml'
            
            startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
                  '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
                  '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="occupants">',
                  '<description> Vulnerability model to calculate fatalities</description>']        
            endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
            
            outFile = open(filename, 'w')
            outFile.writelines("%s\n" % text for text in startText)
            
            # Loop over building classes and occupancy types  
            for i, currentBuildingClass in enumerate(building_classes):   
                         
                for k, currentIM in enumerate(IMT):
                                                
                    if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                            
                        for l, currentOccupancy in enumerate(OCC):

                            # Define general information
                            imt         = IMT_XML[k]
                            ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                               
                            # Load structural DS4 input files
                            thetas = efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 3:7].values.tolist()
                            betas  = [efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 7]]*4                              
                            
                            # Load the consequence model
                            currentConsequence = consequence_df.iloc[consequence_df[consequence_df.taxonomy == ID].index[0], 3:7].values.tolist()
                            
                            array = calculate_structural_vulnerability(thetas, betas, currentConsequence)
                            loss_ratios = array[:,1]
                            
                            # Perform the calculation to write to xml               
                            cov_loss=np.zeros(loss_ratios.shape)               
                            for m in range(loss_ratios.shape[0]):                        
                                mean_loss_ratio=loss_ratios[m]
                                if mean_loss_ratio<1e-4:
                                    loss_ratios[m]=1e-8
                                    cov_loss[m] = 1e-8
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss_ratios[m]= 0.99999
                                    cov_loss[m] = 1e-8
                                else:                                  
                                    sigma_loss, _, _ = calculate_sigma_loss(mean_loss_ratio)
                                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                    cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                        
                            outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                            outFile.write('<imls imt="%s" > ' % imt)
                            outFile.writelines('%g ' % iml for iml in INTENSITIES)
                            outFile.write(' </imls>\n')
                            outFile.write('<meanLRs>')
                            outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                            outFile.write(' </meanLRs>\n')
                            outFile.write('<covLRs>')
                            outFile.writelines('%g ' % cv for cv in cov_loss)
                            outFile.write(' </covLRs>\n')
                            outFile.write('</vulnerabilityFunction>\n')
                    
                    else:
                        
                        pass
                               
            # Close off the xml file 
            outFile.writelines('%s' % text for text in endText)    
            outFile.close()
            
    elif consequenceType == 'AREA':
                   
        # Define filename to print xml to 
        filename    = f'{vulnerability_out}/vulnerability_area.xml'
        
        startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
              '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
              '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="area">',
              '<description> Complete damage state written as vulnerability used for area loss calculation</description>']        
        endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
        
        outFile = open(filename, 'w')
        outFile.writelines("%s\n" % text for text in startText)
        
        # Loop over building classes and occupancy types  
        for i, currentBuildingClass in enumerate(building_classes):   
                     
            for k, currentIM in enumerate(IMT):
                                            
                if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                        
                    for l, currentOccupancy in enumerate(OCC):
                           
                        # Load structural DS4 input files
                        theta_ds4 = efficiency_df[efficiency_df.Building_class == currentBuildingClass].Median_ds4.tolist()[0]  # Get the DS4 (Complete Damage) intensity 
                        beta_ds4  = efficiency_df[efficiency_df.Building_class == currentBuildingClass].Beta.tolist()[0]        # Get the DS4 (Complete damage) dispersion                    
                        p_ds4     = get_fragility_function(INTENSITIES, theta_ds4, beta_ds4)                                                                                                     # Get the DS3 fragility function                                   
                        loss_ratios = p_ds4
        
                        # Define general information
                        imt         = IMT_XML[k]
                        ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                        
                        # Perform the calculation to write to xml               
                        cov_loss=np.zeros(loss_ratios.shape)               
                        for m in range(loss_ratios.shape[0]):                        
                            mean_loss_ratio=loss_ratios[m]
                            if mean_loss_ratio<1e-4:
                                loss_ratios[m]=1e-8
                                cov_loss[m] = 1e-8
                            elif np.abs(1-mean_loss_ratio)<1e-4:
                                loss_ratios[m]= 0.99999
                                cov_loss[m] = 1e-8
                            else:                                  
                                sigma_loss, _, _ = calculate_sigma_loss(mean_loss_ratio)
                                max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                    
                        outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                        outFile.write('<imls imt="%s" > ' % imt)
                        outFile.writelines('%g ' % iml for iml in INTENSITIES)
                        outFile.write(' </imls>\n')
                        outFile.write('<meanLRs>')
                        outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                        outFile.write(' </meanLRs>\n')
                        outFile.write('<covLRs>')
                        outFile.writelines('%g ' % cv for cv in cov_loss)
                        outFile.write(' </covLRs>\n')
                        outFile.write('</vulnerabilityFunction>\n')
                
                else:
                    
                    pass
                           
        # Close off the xml file 
        outFile.writelines('%s' % text for text in endText)    
        outFile.close()
        
    elif consequenceType == 'NUMBER':
        
        # Define filename to print xml to 
        filename    = f'{vulnerability_out}/vulnerability_number.xml'
        
        startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
              '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
              '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="number">',
              '<description> Complete damage state written as vulnerability used for number of damaged buildings calculation</description>']        
        endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
        
        outFile = open(filename, 'w')
        outFile.writelines("%s\n" % text for text in startText)
        
        # Loop over building classes and occupancy types  
        for i, currentBuildingClass in enumerate(building_classes):   
                     
            for k, currentIM in enumerate(IMT):
                                            
                if currentIM in efficiency_df.iloc[efficiency_df[efficiency_df.Building_class == currentBuildingClass].index[0], 2]:
                                        
                    for l, currentOccupancy in enumerate(OCC):
                           
                        # Load structural DS4 input files
                        theta_ds4 = efficiency_df[efficiency_df.Building_class == currentBuildingClass].Median_ds4.tolist()[0]  # Get the DS4 (Complete Damage) intensity 
                        beta_ds4  = efficiency_df[efficiency_df.Building_class == currentBuildingClass].Beta.tolist()[0]        # Get the DS4 (Complete damage) dispersion                    
                        p_ds4     = get_fragility_function(INTENSITIES, theta_ds4, beta_ds4)                                                                                                     # Get the DS3 fragility function                                                           
                        loss_ratios = p_ds4
        
                        # Define general information
                        imt         = IMT_XML[k]
                        ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                        
                        # Perform the calculation to write to xml               
                        if uncert:
                            cov_loss=np.zeros(loss_ratios.shape)               
                            for m in range(loss_ratios.shape[0]):                        
                                mean_loss_ratio=loss_ratios[m]
                                if mean_loss_ratio<1e-4:
                                    loss_ratios[m]=1e-8
                                    cov_loss[m] = 1e-8
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss_ratios[m]= 0.99999
                                    cov_loss[m] = 1e-8
                                else:                                  
                                    sigma_loss, _, _ = calculate_sigma_loss(mean_loss_ratio)
                                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                    cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                                    
                        outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                        outFile.write('<imls imt="%s" > ' % imt)
                        outFile.writelines('%g ' % iml for iml in INTENSITIES)
                        outFile.write(' </imls>\n')
                        outFile.write('<meanLRs>')
                        outFile.writelines('%g ' % lrs for lrs in loss_ratios)
                        outFile.write(' </meanLRs>\n')
                        outFile.write('<covLRs>')
                        outFile.writelines('%g ' % cv for cv in cov_loss)
                        outFile.write(' </covLRs>\n')
                        outFile.write('</vulnerabilityFunction>\n')
                
                else:
                    
                    pass
                           
        # Close off the xml file 
        outFile.writelines('%s' % text for text in endText)    
        outFile.close()

#%%

def write_vulnerability_to_xml(BLDG_CLASSES,
                               EFFICIENCY_DF,
                               VULNERABILITY_IN,
                               VULNERABILITY_OUT,
                               TYPE,
                               UNCERTAINTY):
    
    # Define constants
    IMT     = ['PGA','SA(0.3s)','SA(0.6s)','SA(1.0s)']         # Intensity measure labels
    IMT_XML = ['PGA','SA(0.3)','SA(0.6)','SA(1.0)']            # Intensity measure labels for xml outputs
    OCC     = ['RES','COM','IND']                              # Occupancy Types
    REGIONS = ['']
    # Define filename to print XML to 
    filename    = f'{VULNERABILITY_OUT}/vulnerability_{TYPE}.xml'
    
    # Define the start and end text of the XML file
    if TYPE == 'structural':
    
        startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
              '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
              '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="structural">',
              '<description>structural vulnerability model</description>']        
        endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
        
    elif TYPE == 'nonstructural':
        
        startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
              '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
              '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="nonstructural">',
              '<description>nonstructural vulnerability model</description>']        
        endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']

    elif TYPE == 'contents':
        
        startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
              '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
              '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="contents">',
              '<description>content vulnerability model</description>']        
        endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
    
    elif TYPE == 'building':
        
        startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
              '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
              '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="structural">',
              '<description>building vulnerability model</description>']        
        endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
        
    elif TYPE == 'total':
        
        startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
              '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
              '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="structural">',
              '<description>total vulnerability model</description>']        
        endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']


    outFile = open(filename, 'w')
    outFile.writelines("%s\n" % text for text in startText)

    
    # Loop over building classes and occupancy types  
    for i, currentBuildingClass in enumerate(BLDG_CLASSES):   
                      
        for j, currentIMT in enumerate(IMT):
                                            
            if currentIMT in EFFICIENCY_DF.iloc[EFFICIENCY_DF[EFFICIENCY_DF.Building_class == currentBuildingClass].index[0], 2]:
                                    
                for k, currentOccupancy in enumerate(OCC):
                    

                    if TYPE == 'structural':
                        
                        # Load the file
                        input_file = pd.read_csv(f'{VULNERABILITY_IN}/structural/csv/structural_vulnerability_{currentBuildingClass}_{currentIMT}.csv')
                        
                        # Extract the imls and loss arrays
                        imls = input_file.values[:,0]
                        loss = input_file.values[:,1]
                        
                    elif TYPE == 'nonstructural':
                        
                        # Load the file
                        input_file = pd.read_csv(f'{VULNERABILITY_IN}/nonstructural/csv/nonstructural_vulnerability_{currentBuildingClass}_{currentIMT}_{currentOccupancy}.csv')
                        
                        # Extract the imls and loss arrays
                        imls = input_file.values[:,0]
                        loss = input_file.values[:,1]
                        
                    elif TYPE == 'contents':
                        
                        # Load the file
                        input_file = pd.read_csv(f'{VULNERABILITY_IN}/contents/csv/contents_vulnerability_{currentBuildingClass}_{currentIMT}_{currentOccupancy}.csv')
                        
                        # Extract the imls and loss arrays
                        imls = input_file.values[:,0]
                        loss = input_file.values[:,1]
                    
                    
                    elif TYPE == 'building':
                        
                        # Load the file
                        input_file_1 = pd.read_csv(f'{VULNERABILITY_IN}/structural/csv/structural_vulnerability_{currentBuildingClass}_{currentIMT}_{currentOccupancy}.csv')
                        input_file_2 = pd.read_csv(f'{VULNERABILITY_IN}/nonstructural/csv/nonstructural_vulnerability_{currentBuildingClass}_{currentIMT}_{currentOccupancy}.csv')
                                                
                        # Extract the imls and loss arrays
                        imls   = input_file_1.values[:,0]
                        loss_1 = input_file_1.values[:,1]
                        loss_2 = input_file_2.values[:,1]
                        
                        if currentOccupancy == 'RES':
                            sv  = 3.0/8.0 
                            nsv = 5.0/8.0
                        elif currentOccupancy == 'COM':
                            sv = 2.0/5.0
                            nsv = 3.0/5.0
                        elif currentOccupancy == 'IND':
                            sv  = 3.0/8.0 
                            nsv = 5.0/8.0
                        
                        loss = np.zeros((loss_1.shape[0],1)) 
                        loss = sv*loss_1 + nsv*loss_2


                    elif TYPE == 'total':
                        
                        # Load the file
                        input_file_1 = pd.read_csv(f'{VULNERABILITY_IN}/structural/csv/structural_vulnerability_{currentBuildingClass}_{currentIMT}_{currentOccupancy}.csv')
                        input_file_2 = pd.read_csv(f'{VULNERABILITY_IN}/nonstructural/csv/nonstructural_vulnerability_{currentBuildingClass}_{currentIMT}_{currentOccupancy}.csv')
                        input_file_3 = pd.read_csv(f'{VULNERABILITY_IN}/contents/csv/contents_vulnerability_{currentBuildingClass}_{currentIMT}_{currentOccupancy}.csv')
                                                
                        # Extract the imls and loss arrays
                        imls   = input_file_1.values[:,0]
                        loss_1 = input_file_1.values[:,1]
                        loss_2 = input_file_2.values[:,1]
                        loss_3 = input_file_3.values[:,1]
                        
                        if currentOccupancy() == 'RES':
                            sv  = 0.30 
                            nsv = 0.50
                            cv  = 0.20
                        elif currentOccupancy() == 'COM':
                            sv  = 0.20 
                            nsv = 0.30
                            cv  = 0.50
                        elif currentOccupancy() == 'IND':
                            sv  = 0.15 
                            nsv = 0.25
                            cv  = 0.60 
                            
                        loss = np.zeros((loss_1.shape[0],1))
                        loss = sv*loss_1 + nsv*loss_2 + cv*loss_3
                                                
                    # Define general information
                    imt         = IMT_XML[j]
                    ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                    
                    # Perform the calculation to write to xml               
                    if UNCERTAINTY:
                        cov_loss=np.zeros(loss.shape)               
                        for m in range(loss.shape[0]):                        
                            mean_loss_ratio=loss[m]
                            if mean_loss_ratio<1e-4:
                                loss[m]=1e-8
                                cov_loss[m] = 1e-8
                            elif np.abs(1-mean_loss_ratio)<1e-4:
                                loss[m]= 0.99999
                                cov_loss[m] = 1e-8
                            else:                                  
                                sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                    else:
                        cov_loss=np.zeros(loss.shape)               
                        for m in range(loss.shape[0]):                        
                            mean_loss_ratio=loss[m]
                            if mean_loss_ratio<1e-4:
                                loss[m]=1e-8
                                cov_loss[m] = 0.0
                            elif np.abs(1-mean_loss_ratio)<1e-4:
                                loss[m]= 0.99999
                                cov_loss[m] = 0.0
                            else:                                  
                                cov_loss[m] = 0.0
                                                
                    outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                    outFile.write('<imls imt="%s" > ' % imt)
                    outFile.writelines('%g ' % iml for iml in imls)
                    outFile.write(' </imls>\n')
                    outFile.write('<meanLRs>')
                    outFile.writelines('%g ' % lrs for lrs in loss)
                    outFile.write(' </meanLRs>\n')
                    outFile.write('<covLRs>')
                    outFile.writelines('%g ' % cv for cv in cov_loss)
                    outFile.write(' </covLRs>\n')
                    outFile.write('</vulnerabilityFunction>\n')
                                   
    # Close off the xml file 
    outFile.writelines('%s' % text for text in endText)    
    outFile.close()

        
def write_vulnerability_singleIM_to_xml(BLDG_CLASSES,
                                        EFFICIENCY_DF,
                                        VULNERABILITY_IN,
                                        VULNERABILITY_OUT,
                                        TYPE,
                                        UNCERTAINTY):
    
    # Define constants
    IMT     = ['PGA','SA(0.3s)','SA(0.6s)','SA(1.0s)']         # Intensity measure labels
    IMT_XML = ['PGA','SA(0.3)','SA(0.6)','SA(1.0)']            # Intensity measure labels for xml outputs
    OCC     = ['RES','COM','IND']                              # Occupancy Types
    REGIONS = ['Africa',
               'Caribbean_Central_America',
               'Central_Asia',
               'East_Asia',
               'Europe',
               'Middle_East',
               'North_America',
               'North_Asia',
               'Oceania',
               'South_America',
               'South_Asia',
               'SouthEast_Asia']                               # GRM regions 

    for _, currentRegion in enumerate(REGIONS):
        
        XML_output = f'{VULNERABILITY_OUT}/vulnerability/{currentRegion}'
        if not os.path.exists(f'{XML_output}'):
            os.makedirs(f'{XML_output}')
    
        # Define filename to print XML to 
        filename    = f'{XML_output}/vulnerability_{TYPE}.xml'
        
        # Define the start and end text of the XML file
        if TYPE == 'structural':
        
            startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
                  '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
                  '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="structural">',
                  '<description>structural vulnerability model</description>']        
            endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
            
        elif TYPE == 'nonstructural':
            
            startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
                  '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
                  '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="nonstructural">',
                  '<description>nonstructural vulnerability model</description>']        
            endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
    
        elif TYPE == 'contents':
            
            startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
                  '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
                  '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="contents">',
                  '<description>content vulnerability model</description>']        
            endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
        
        elif TYPE == 'building':
            
            startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
                  '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
                  '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="structural">',
                  '<description>building vulnerability model</description>']        
            endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
            
        elif TYPE == 'total':
            
            startText = ['<?xml version="1.0" encoding="UTF-8"?>', 
                  '<nrml xmlns="http://openquake.org/xmlns/nrml/0.5">',
                  '<vulnerabilityModel id="vulnerability_model" assetCategory="buildings" lossCategory="structural">',
                  '<description>total vulnerability model</description>']        
            endText = ['\n</vulnerabilityModel>\n', '</nrml>\n']
    
    
        outFile = open(filename, 'w')
        outFile.writelines("%s\n" % text for text in startText)
    
        
        # Loop over building classes and occupancy types  
        for i, currentBuildingClass in enumerate(BLDG_CLASSES):   
                          
            for j, currentIMT in enumerate(IMT):
                                                
                if currentIMT in EFFICIENCY_DF.iloc[EFFICIENCY_DF[EFFICIENCY_DF.Building_class == currentBuildingClass].index[0], 2]:
                                        
                    for k, currentOccupancy in enumerate(OCC):
                        
    
                        if TYPE == 'structural':
                            
                            # Load the file
                            input_file = pd.read_csv(f'{VULNERABILITY_IN}/structural/csv/structural_vulnerability_{currentBuildingClass}_{currentIMT}.csv')
                            
                            # Extract the imls and loss arrays
                            imls = input_file.values[:,0]
                            loss = input_file.values[:,1]
                            
                        elif TYPE == 'nonstructural':
                            
                            # Load the file
                            input_file = pd.read_csv(f'{VULNERABILITY_IN}/nonstructural/csv/nonstructural_vulnerability_{currentBuildingClass}_{currentIMT}_{currentOccupancy}.csv')
                            
                            # Extract the imls and loss arrays
                            imls = input_file.values[:,0]
                            loss = input_file.values[:,1]
                            
                        elif TYPE == 'contents':
                            
                            # Load the file
                            input_file = pd.read_csv(f'{VULNERABILITY_IN}/contents/csv/contents_vulnerability_{currentBuildingClass}_{currentIMT}_{currentOccupancy}.csv')
                            
                            # Extract the imls and loss arrays
                            imls = input_file.values[:,0]
                            loss = input_file.values[:,1]
                        
                        
                        elif TYPE == 'building':
                            
                            # Load the file
                            input_file_1 = pd.read_csv(f'{VULNERABILITY_IN}/structural/csv/structural_vulnerability_{currentBuildingClass}_{currentIMT}_{currentOccupancy}.csv')
                            input_file_2 = pd.read_csv(f'{VULNERABILITY_IN}/nonstructural/csv/nonstructural_vulnerability_{currentBuildingClass}_{currentIMT}_{currentOccupancy}.csv')
                                                    
                            # Extract the imls and loss arrays
                            imls   = input_file_1.values[:,0]
                            loss_1 = input_file_1.values[:,1]
                            loss_2 = input_file_2.values[:,1]
                            
                            if currentOccupancy == 'RES':
                                sv  = 3.0/8.0 
                                nsv = 5.0/8.0
                            elif currentOccupancy == 'COM':
                                sv = 2.0/5.0
                                nsv = 3.0/5.0
                            elif currentOccupancy == 'IND':
                                sv  = 3.0/8.0 
                                nsv = 5.0/8.0
                            
                            loss = np.zeros((loss_1.shape[0],1)) 
                            loss = sv*loss_1 + nsv*loss_2
    
    
                        elif TYPE == 'total':
                            
                            # Load the file
                            input_file_1 = pd.read_csv(f'{VULNERABILITY_IN}/structural/csv/structural_vulnerability_{currentBuildingClass}_{currentIMT}_{currentOccupancy}.csv')
                            input_file_2 = pd.read_csv(f'{VULNERABILITY_IN}/nonstructural/csv/nonstructural_vulnerability_{currentBuildingClass}_{currentIMT}_{currentOccupancy}.csv')
                            input_file_3 = pd.read_csv(f'{VULNERABILITY_IN}/contents/csv/contents_vulnerability_{currentBuildingClass}_{currentIMT}_{currentOccupancy}.csv')
                                                    
                            # Extract the imls and loss arrays
                            imls   = input_file_1.values[:,0]
                            loss_1 = input_file_1.values[:,1]
                            loss_2 = input_file_2.values[:,1]
                            loss_3 = input_file_3.values[:,1]
                            
                            if currentOccupancy() == 'RES':
                                sv  = 0.30 
                                nsv = 0.50
                                cv  = 0.20
                            elif currentOccupancy() == 'COM':
                                sv  = 0.20 
                                nsv = 0.30
                                cv  = 0.50
                            elif currentOccupancy() == 'IND':
                                sv  = 0.15 
                                nsv = 0.25
                                cv  = 0.60 
                                
                            loss = np.zeros((loss_1.shape[0],1))
                            loss = sv*loss_1 + nsv*loss_2 + cv*loss_3
                                                    
                        # Define general information
                        imt         = IMT_XML[j]
                        ID          = currentBuildingClass.replace('_','/')+f'/{currentOccupancy}'
                        
                        # Perform the calculation to write to xml               
                        if UNCERTAINTY:
                            cov_loss=np.zeros(loss.shape)               
                            for m in range(loss.shape[0]):                        
                                mean_loss_ratio=loss[m]
                                if mean_loss_ratio<1e-4:
                                    loss[m]=1e-8
                                    cov_loss[m] = 1e-8
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss[m]= 0.99999
                                    cov_loss[m] = 1e-8
                                else:                                  
                                    sigma_loss = np.sqrt(mean_loss_ratio*(-0.7-2*mean_loss_ratio+np.sqrt(6.8*mean_loss_ratio+0.5)))
                                    max_sigma = np.sqrt(mean_loss_ratio*(1-mean_loss_ratio))
                                    sigma_loss_ratio = np.min([max_sigma, sigma_loss])
                                    cov_loss[m] = np.min([sigma_loss_ratio/mean_loss_ratio, 0.98*max_sigma/mean_loss_ratio])
                        else:
                            cov_loss=np.zeros(loss.shape)               
                            for m in range(loss.shape[0]):                        
                                mean_loss_ratio=loss[m]
                                if mean_loss_ratio<1e-4:
                                    loss[m]=1e-8
                                    cov_loss[m] = 0.0
                                elif np.abs(1-mean_loss_ratio)<1e-4:
                                    loss[m]= 0.99999
                                    cov_loss[m] = 0.0
                                else:                                  
                                    cov_loss[m] = 0.0
                                                    
                        outFile.write('\n<vulnerabilityFunction id="%s" dist="BT">\n' % ID)      
                        outFile.write('<imls imt="%s" > ' % imt)
                        outFile.writelines('%g ' % iml for iml in imls)
                        outFile.write(' </imls>\n')
                        outFile.write('<meanLRs>')
                        outFile.writelines('%g ' % lrs for lrs in loss)
                        outFile.write(' </meanLRs>\n')
                        outFile.write('<covLRs>')
                        outFile.writelines('%g ' % cv for cv in cov_loss)
                        outFile.write(' </covLRs>\n')
                        outFile.write('</vulnerabilityFunction>\n')
                                       
        # Close off the xml file 
        outFile.writelines('%s' % text for text in endText)    
        outFile.close()

