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


