### Import libraries
import os
import re
import pickle
import numpy as np


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
