### Import libraries
import os
import re
import sys
import shutil
import zipfile
import requests
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path
from scipy.io import loadmat
from scipy.stats import skew
from selenium import webdriver
from time import gmtime, sleep
import matplotlib.pyplot as plt
from scipy import interpolate, integrate
from matplotlib.ticker import ScalarFormatter, NullFormatter
from .webdriverdownloader import ChromeDriverDownloader, GeckoDriverDownloader
from .recordselector_utils import get_esm_token, content_from_zip, read_nga, read_esm, random_multivariate_normal

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

plt.rc('font', size=FONTSIZE_3)  # controls default text sizes
plt.rc('axes', titlesize=FONTSIZE_3)  # fontsize of the axes title
plt.rc('axes', labelsize=FONTSIZE_2)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONTSIZE_3)  # fontsize of the tick labels
plt.rc('ytick', labelsize=FONTSIZE_3)  # fontsize of the tick labels
plt.rc('legend', fontsize=FONTSIZE_2)  # legend fontsize
plt.rc('figure', titlesize=FONTSIZE_1)  # fontsize of the figure title


class recordSelector:
    
    def __init__(self, oq_ini_path, oq_out_path, export_path=None, pflag=True, database=None):
        """
        Initialise the RecordSelector with OpenQuake outputs and a ground motion database.
        
        Parameters
        ----------
        oq_ini_path : str or Path
            Path to OpenQuake job.ini file.
        oq_out_path : str or Path
            Path to OpenQuake output directory.
        export_path : str or Path, optional
            Directory where figures/results will be saved. Defaults to same as oq_ini_path parent.
        pflag : bool, optional
            Whether to show plots interactively.
        database : str or dict, optional
            Either:
            - The name of the database (e.g., 'NGA_W2')
            - A full path to the database file
            - A preloaded Python dictionary containing spectral data
        """
        
        # --- Basic paths ---
        self.oq_ini_path = Path(oq_ini_path)
        self.oq_out_path = Path(oq_out_path)
        self.export_path = Path(export_path or self.oq_ini_path.parent)
        self.figure_path = self.export_path / 'figures'   # Output path for figures
        self.pflag = pflag
        
        # --- Create export paths ---
        self.export_path.mkdir(parents=True, exist_ok=True)
        self.figure_path.mkdir(parents=True, exist_ok=True)

        # --- Internal State for Plotting/Processing ---
        self.hz_results = {} # Stores plotting data for all locations (raw only)
        self.inv_t = None
        self.period_range = None
        
        # --- Validate database input ---
        if database is None:
            raise ValueError("A valid ground motion database (path or dict) must be provided.")
        
        # --- Load database (simplified) ---
        meta_data_dir = Path(__file__).resolve().parent / "metadata"
        if not str(database).endswith(".mat"):
            database = f"{database}.mat"        
        matfile = meta_data_dir / database


        if not matfile.exists():
            raise FileNotFoundError(f"Database file not found: {matfile}")
        self.database = loadmat(matfile, squeeze_me=True)
        self.database['Name'] = Path(database).stem

                
        # --- Initialise results containers ---
        self.results = {"hazard": {}, "conditional_spectrum": {}}
        self.target_spectra = {}
        self.im_Tstar = {}


    #----------------------------------------------------------------------------#
    #                                                                            #        
    #          OpenQuake Conditional Spectrum Calc Post-Processing Functions     #
    #                                                                            #
    #----------------------------------------------------------------------------#
    def _process_hazard_curves(self, oq_out_path):
        """
        Process OpenQuake hazard curve CSV files, clean the data.
        Stores raw plotting data for all locations.

        Parameters
        ----------
        oq_out_path : str or Path
            Directory containing hazard_curve-mean-*.csv files.

        Returns
        -------
        all_results : dict
            Dictionary keyed by location "(lon, lat)", with structure:
                results["(lon, lat)"][im_type] = {"imls": iml_vals_clean, "poes": poe_vals_clean}
            where iml_vals_clean and poe_vals_clean are numpy arrays.
        """
        
        results_dir = self.oq_out_path
        file_prefix = 'hazard_curve-mean'
        hazard_files = sorted(results_dir.glob(f"{file_prefix}-*.csv"))
        
        os.makedirs(results_dir, exist_ok=True)
        
        all_results = {}
        
        if not hazard_files:
            raise FileNotFoundError(f"No {file_prefix}-*.csv files found in {results_dir}.")

        # Data structures to hold information for ALL locations, used for plotting later
        hz_results = {}
        representative_loc_key = None # To track the key for common file saving
        inv_t = None
        
        # --- 1. Data Processing Loop ---
        for file_path in hazard_files:
            file_name = file_path.name

            # --- Extract Metadata (IM type, ID, and Inv Time) ---
            try:
                im_type = (file_name.rsplit('-', 2)[-1]).rsplit('_', 1)[0]
                idn = (file_name.rsplit('_', 1)[-1]).rsplit('.')[0]
            except IndexError:
                print(f"Warning: Skipping file {file_name} due to unexpected naming format.")
                continue

            df = pd.read_csv(file_path, skiprows=1)

            with open(file_path, "r") as f:
                header_line = f.readline().split(',')
                inv_t_str = next(filter(lambda x: 'investigation_time=' in x, header_line), None)
                if inv_t_str:
                    inv_t = float(inv_t_str.replace(" investigation_time=", ""))
                else:
                    print(f"Warning: Could not determine investigation time for {file_name}. Skipping.")
                    continue
            
            iml_values = np.array([float(i[4:]) for i in df.columns.values[3:]])

            # --- Process Each Site ---
            for site_index, row in df.iterrows():
                lon, lat = row["lon"], row["lat"]  
                loc_key = f"({lon}, {lat})"
                
                poe_values = row.iloc[3:].values.astype(float)
                
                mask = ~(np.isnan(poe_values) | np.isinf(poe_values))
                poe_vals_clean = poe_values[mask]
                iml_vals_clean = iml_values[mask]
                
                if len(poe_vals_clean) < 2:
                    continue
                
                # Set the representative key if not set yet (used for saving common files)
                if representative_loc_key is None:
                    representative_loc_key = loc_key
                                
                # --- 2. Store the Results (Raw Data) and Plotting Data ---
                all_results.setdefault(loc_key, {})
                all_results[loc_key][im_type] = {"imls": iml_vals_clean, "poes": poe_vals_clean}
                
                # Store plotting data for this specific location
                hz_results.setdefault(loc_key, {})
                hz_results[loc_key][im_type] = {'raw': (iml_vals_clean, poe_vals_clean),
                                                      'id_no': idn}
                
        # Store data needed for plotting on self
        self.hz_results = hz_results
        self.inv_t = inv_t

        return all_results
    
    def _process_conditional_spectra(self, oq_out_dir):
        """
        Process multiple OpenQuake conditional spectrum CSV files.
        Each file corresponds to a single site (lon, lat).
        """
        
        oq_out_dir = Path(oq_out_dir)
        if not oq_out_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {oq_out_dir}")
        
        cs_files = sorted(oq_out_dir.glob("conditional-spectrum-*.csv"))
        if not cs_files:
            # Not an error if no CS files are found, just return empty
            return {}
        
        # Initialise the conditional spectrum results dictionary
        cs_results = {}
        
        for cs_file in cs_files:
            # --- 1. Extract IMT from filename (e.g., SA(1.0) from conditional-spectrum-0.00-SA(1.0).csv) ---
            imt_match = re.search(r"conditional-spectrum-[\d\.]+-([\w\(\)\.]+)\.csv$", cs_file.name)
            imt_key = imt_match.group(1) if imt_match else "SA_ref"
            
            # --- 2. Parse metadata from header line ---
            with open(cs_file, "r") as f:
                header_line = f.readline().strip()
                
            lon, lat = None, None
            for part in header_line.split(","):
                
                part = part.strip()  # <--- strip spaces
                
                # Parse the longitude
                if "lon=" in part:
                    try:
                        lon = str(part.split("=")[1])
                    except ValueError:
                        pass
                # Parse the latitude
                elif "lat=" in part:
                    try:
                        lat = str(part.split("=")[1].strip("'\" "))
                    except ValueError:
                        pass
            
            # Raise error if longitude, latitude pair are not properly extracted
            if lon is None or lat is None:
                raise ValueError(f"Could not extract lon/lat from header of {cs_file.name}")
        
            loc_key = f"({str(lon)}, {str(lat)})"
            cs_results.setdefault(loc_key, {})
            # Store results under the IMT key
            cs_results[loc_key].setdefault(imt_key, {}) 

            # --- 3. Read data (skip header line) ---
            df = pd.read_csv(cs_file, skiprows=1)
            df.columns = [c.strip().lower() for c in df.columns]
        
            # Identify columns automatically
            col_poe      = next(c for c in df.columns if "poe" in c)
            col_period   = next(c for c in df.columns if "period" in c)
            col_mean     = next(c for c in df.columns if "mea" in c)
            col_std      = next(c for c in df.columns if "std" in c)
        
            # --- 4. Process each POE group ---
            for poe, group in df.groupby(col_poe):
                            
                # Note: Assuming inv_t of 50 years based on common usage with this formula
                return_period = round(-50.0 / np.log(1 - poe))
                period        = group[col_period].to_numpy()
                mean          = group[col_mean].to_numpy()
                sigma_ln_oq   = group[col_std].to_numpy() # OQ output 'std' is sigma_ln
        
                # Compute lognormal uncertainty bounds (consistent with OpenQuake)
                upper = mean * np.exp(sigma_ln_oq)
                lower = mean * np.exp(-sigma_ln_oq)
                
                # Pack the results
                cs_results[loc_key][imt_key][return_period] = {
                    "period": period,
                    "mean": mean,
                    "mean+std": upper,
                    "mean-std": lower,
                    "sigma_ln": sigma_ln_oq}
                
                # Store the period range to attributes
                self.period_range = period
                
        return cs_results

    
    def _plot_oq_cs_calc(self, results):
        """Generates a figure with hazard curves and conditional spectra subplots for EACH location."""

        hz_results = self.hz_results
        inv_t = self.inv_t
        cs_results = results.get("conditional_spectrum", {})
        
        # Get all unique location keys from both hazard and conditional spectra results
        all_loc_keys = set(hz_results.keys()).union(set(cs_results.keys()))
        
        if not all_loc_keys:
             print("No location keys found in results for plotting.")
             return

        for loc_key in all_loc_keys:
            # Data for the current location
            current_hazard_data = hz_results.get(loc_key, {})
            current_cs_data = cs_results.get(loc_key, {})
            
            # Skip if no relevant data for the current key
            if not current_hazard_data and not current_cs_data:
                continue

            try:
                lon, lat = loc_key.strip('()').split(', ')
            except ValueError:
                print(f"Skipping plot for invalid key format: {loc_key}")
                continue
            
            # --- 1. Setup Figure ---
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f"OQ Conditional Spectrum Calculation Results @ Lon: {lon}, Lat: {lat}", fontsize=14)

            # --- 2. Plot Hazard Curves (Left Subplot) ---
            ax_haz = axes[0]
            
            if current_hazard_data and inv_t is not None:
                
                for im_type, data in current_hazard_data.items():
                                        
                    # Plot hazard curves
                    iml_raw, poe_raw = data['raw']
                    ax_haz.loglog(iml_raw, poe_raw, linestyle='-', label=f'{im_type}')
                    
                ax_haz.set_xlabel('Intensity Measure Level [g]')
                ax_haz.set_ylabel(f'Probability of Exceedance in {inv_t:.0f} years')
                ax_haz.set_title('Mean Hazard Curves')
                #ax_haz.legend()
                ax_haz.grid(True, which="both", ls="--", alpha=0.5)
                ax_haz.set_xlim([5e-2, 5e0])
                ax_haz.set_ylim([1e-5, 1e0])
            else:
                 ax_haz.set_title('Hazard Curve Data Not Available')


            # --- 3. Plot Conditional Spectra (Right Subplot) ---
            ax_cs = axes[1]
            
            if current_cs_data:
                # Conditional Spectra results are stored under Location -> IMT -> RP -> Data
                # Find the first conditioning IMT for plotting
                first_imt = next(iter(current_cs_data.keys()))
                rp_data_dict = current_cs_data[first_imt]
                
                # Sort by return period for clear plotting order
                sorted_rps = sorted(rp_data_dict.keys())
                
                for rp in sorted_rps:
                    data = rp_data_dict[rp]
                    periods = data["period"]
                    mean = data["mean"]
                    upper = data["mean+std"] # Use pre-calculated bounds
                    lower = data["mean-std"] 
                    
                    # Plot Mean Spectrum
                    line, = ax_cs.loglog(periods, mean, label=f'RP: {rp} yrs')
                    
                    # Plot Mean +/- 1 Sigma
                    ax_cs.fill_between(periods, lower, upper, alpha=0.1, color=line.get_color())
                    
                ax_cs.set_xlabel('Period (T) [s]')
                ax_cs.set_ylabel('Pseudo-Spectral Acceleration (SA) [g]')
                ax_cs.set_title(r'Conditional Spectra $\mu \pm 1\sigma$')
                ax_cs.legend(loc='lower left', fontsize = 10)
                ax_cs.grid(True, which="both", ls="--", alpha=0.5)
                ax_cs.set_xscale('log')
                ax_cs.set_yscale('log')
            else:
                ax_cs.set_title('Conditional Spectra Data Not Available')
            
            # --- 4. Finalize and Save Plot ---
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
            
            # Save using the loc_key in the filename
            # Replace parentheses and comma-space with underscore for safe filename
            safe_loc_key = loc_key.strip('()').replace(', ', '_').replace('.', 'p').replace('-', 'm') 
            fname_plot = self.figure_path / f'Hazard_and_CS_Plots_{safe_loc_key}.png'
            plt.savefig(fname_plot, format='png', dpi=300)
            
            if self.pflag:
                plt.show()
            
            plt.close(fig)


    def process_oq_cs_calc(self):
        """
        Process OpenQuake hazard + conditional spectra and populate self.results.
        Also extracts Tstar from imt_ref and computes im_Tstar for each conditional spectrum.
        """
        if not self.oq_ini_path.exists():
            raise FileNotFoundError(f"OpenQuake ini file not found: {self.oq_ini_path}")
        oq_model_dir = self.oq_ini_path.parent
        
        # Read job.ini parameters (only need imt_ref now)
        poes, results_dir = [], None
        Tstar = None
        with open(self.oq_ini_path, "r") as f:
            for line in f:
                line = line.strip()
                # Remove reading of poes
                if line.startswith("export_dir"):
                    results_dir = oq_model_dir / line.split("=")[1].strip()
                elif line.startswith("imt_ref"):
                    if "SA(" in line:
                        match = re.search(r"SA\(([\d\.]+)\)", line)
                        if match:
                            Tstar = float(match.group(1))
                    elif "PGA" in line:
                        Tstar = 0.01  # assign small period for PGA
        
        if not results_dir: # poes is no longer mandatory
            raise ValueError("Missing required parameter in job.ini (export_dir).")
        
        if Tstar is None:
            print("imt_ref not found in ini file. Skipping Tstar assignment.")
        else:
            self.Tstar = np.array([Tstar])  # store as array for consistency
        
        # Process hazard curves (stores plotting data on self, without poes argument)
        hazard_results = self._process_hazard_curves(self.oq_out_path)
        self.results["hazard"].update(hazard_results)
        
        # Process conditional spectra
        cs_results = self._process_conditional_spectra(self.oq_out_path)
        self.results["conditional_spectrum"].update(cs_results)
        
        # Compute im_Tstar for each location and return period
        if Tstar is not None:
            # cs_results structure: loc_key -> imt_key -> return_period -> data
            for loc_key, imt_dict in cs_results.items():
                for imt_key, rp_dict in imt_dict.items():
                    for return_period, data in rp_dict.items():
                        # Interpolate to get spectral acceleration at Tstar
                        periods = data["period"]
                        mean_spectrum = data["mean"]
                        f_interp = interpolate.interp1d(periods, mean_spectrum, bounds_error=False, fill_value="extrapolate")
                        im_Tstar = f_interp(Tstar)
                        
                        # Store im_Tstar in the data dict
                        data["im_Tstar"] = im_Tstar
                        
                        # Store im_Tstar in self.im_Tstar for general use
                        self.im_Tstar.setdefault(loc_key, {})
                        self.im_Tstar[loc_key].setdefault(imt_key, {})
                        self.im_Tstar[loc_key][imt_key][return_period] = im_Tstar


        # Store updated conditional spectra back to target
        self.target_spectra = cs_results
        
        # --- PLOTTING ---
        if self.pflag: 
            self._plot_oq_cs_calc(self.results)

        return self.results


    #----------------------------------------------------------------------------#
    #                                                                            #        
    #          OpenQuake Conditional Spectrum Calc Post-Processing Functions     #
    #                                                                            #
    #----------------------------------------------------------------------------#
    def _search_global_flatfile(self):
        """
        Searches the database and filters records based on metadata and periods.
        Crucially, it requires all periods in self.periods to be found in the database.
        
        Returns
        -------
        sample_big : numpy.ndarray (IMLs filtered by metadata and truncated periods)
        vs30 : numpy.ndarray
        magnitude : numpy.ndarray
        rjb : numpy.ndarray
        mechanism : numpy.ndarray
        nga_num : numpy.ndarray (None if not NGA_W2)
        eq_id : numpy.ndarray
        station_code : numpy.ndarray (None if not ESM_2018)
        """
        
        # Initialize database-specific variables to None
        nga_num = None
        station_code = None
        filename2 = None # Initialize filename2 as None for num_components == 1 case
        
        # --- 1. Compile Sa and Metadata based on Component Selection (Code remains unchanged) ---
        if self.num_components == 1: 
            sa_known = np.append(self.database['Sa_1'], self.database['Sa_2'], axis=0)
            vs30 = np.append(self.database['soil_Vs30'], self.database['soil_Vs30'], axis=0)
            magnitude = np.append(self.database['magnitude'], self.database['magnitude'], axis=0)
            rjb = np.append(self.database['Rjb'], self.database['Rjb'], axis=0)
            mechanism = np.append(self.database['mechanism'], self.database['mechanism'], axis=0)
            eq_id = np.append(self.database['EQID'], self.database['EQID'], axis=0)
            
            if self.database['Name'] == "NGA_W2":
                nga_num = np.append(self.database['NGA_num'], self.database['NGA_num'], axis=0)
            elif self.database['Name'] == "ESM_2018":
                station_code = np.append(self.database['station_code'], self.database['station_code'], axis=0)
            elif self.database['Name'] == "Global":
                station_code = np.append(self.database['station_code'], self.database['station_code'], axis=0)
                
        elif self.num_components == 2:
            if self.spectrum_definition == 'GeoMean':
                sa_known = np.sqrt(self.database['Sa_1'] * self.database['Sa_2'])
            elif self.spectrum_definition == 'SRSS':
                sa_known = np.sqrt(self.database['Sa_1'] ** 2 + self.database['Sa_2'] ** 2)
            elif self.spectrum_definition == 'ArithmeticMean':
                sa_known = (self.database['Sa_1'] + self.database['Sa_2']) / 2
            elif self.spectrum_definition == 'RotD50':
                sa_known = self.database['Sa_RotD50']
            elif self.spectrum_definition == 'RotD100':
                sa_known = self.database['Sa_RotD100']
            else:
                raise ValueError('Unexpected Sa definition, exiting...')
            
            vs30 = self.database['soil_Vs30']
            magnitude = self.database['magnitude']
            rjb = self.database['Rjb']
            mechanism = self.database['mechanism']
            eq_id = self.database['EQID']
            
            if self.database['Name'] == "NGA_W2":
                nga_num = self.database['NGA_num']
            elif self.database['Name'] == "ESM_2018":
                station_code = self.database['station_code']
            elif self.database['Name'] == "Global":
                station_code = self.database['station_code']

        else:
            raise ValueError('Selection can only be performed for one or two components at the moment, exiting...')
        
        # --- 2. Filtering Records (Only Sa <= 0 filter remains) ---
        
        # Sa cannot be negative or zero. Identify records to be excluded.
        not_allowed = np.unique(np.where(sa_known <= 0)[0]).tolist()
        
        # NOTE: Filtering by vs30_limits, mag_limits, rjb_limits, and mech_limits has been removed.
        
        # Finalize list of allowed indices
        not_allowed = (list(set(not_allowed)))
        allowed = [i for i in range(sa_known.shape[0])]
        for i in not_allowed:
            if i in allowed:
                allowed.remove(i)
        
        # --- 3. Apply Filter and Final Checks ---
        
        # Apply the filter to all data arrays
        sa_known = sa_known[allowed, :]
        vs30 = vs30[allowed]
        magnitude = magnitude[allowed]
        rjb = rjb[allowed]
        mechanism = mechanism[allowed]
        eq_id = eq_id[allowed]
        
        # Handle filename2 filtering
        if self.num_components == 2:
            filename2 = filename2[allowed]
        else:
            filename2 = None

        # Handle database-specific metadata filtering
        if self.database['Name'] == "NGA_W2":
            if nga_num is not None: nga_num = nga_num[allowed]
            station_code = None
        elif self.database['Name'] == "ESM_2018":
            nga_num = None
            if station_code is not None: station_code = station_code[allowed]
        elif self.database['Name'] == "Global":
            nga_num = None
            if station_code is not None: station_code = station_code[allowed]
        
        # --- Period Matching (Ensure periods match the database exactly) ---
        
        record_periods = []
        
        # This loop attempts to find the index of every period in self.periods within the database periods.
        try:
            for period in self.periods:
                # Find the index of the period in the database periods array
                # [0][0] extracts the single index value from the result (np.array([index]))
                match_index = np.where(self.database['Periods'] == period)[0][0] 
                record_periods.append(match_index)
                
        except IndexError as e:
            # Catching the error that occurs when np.where returns an empty array, meaning a period is missing.
            raise ValueError(f"Target period {period} not found in database periods. All target periods must exist in the database.") from e
        
        # Final filter: select only the columns (periods) corresponding to the retained periods
        sample_big = sa_known[:, record_periods]
        
        # Check for invalid input (NaNs)
        if np.any(np.isnan(sample_big)):
            raise ValueError('NaNs found in input response spectra after period filtering.')
        
        # Check if enough records satisfy the criteria
        if self.num_records > len(eq_id):
            raise ValueError('There are not enough records which satisfy',
                            'the given record selection criteria...',
                            'Please use broaden your selection criteria...')
        
        # Return the filtered data
        return sample_big, vs30, magnitude, rjb, mechanism, nga_num, eq_id, station_code


       
    def _search_database(self):
        """
        Searches the database and filters records based on metadata and periods.
        Crucially, it requires all periods in self.periods to be found in the database.
        
        Returns
        -------
        sample_big : numpy.ndarray (IMLs filtered by metadata and truncated periods)
        vs30 : numpy.ndarray
        magnitude : numpy.ndarray
        rjb : numpy.ndarray
        mechanism : numpy.ndarray
        filename1 : numpy.ndarray
        filename2 : numpy.ndarray (None if num_components == 1)
        nga_num : numpy.ndarray (None if not NGA_W2)
        eq_id : numpy.ndarray
        station_code : numpy.ndarray (None if not ESM_2018)
        """
        
        # Initialize database-specific variables to None
        nga_num = None
        station_code = None
        filename2 = None # Initialize filename2 as None for num_components == 1 case
        
        # --- 1. Compile Sa and Metadata based on Component Selection (Code remains unchanged) ---
        if self.num_components == 1: 
            sa_known = np.append(self.database['Sa_1'], self.database['Sa_2'], axis=0)
            vs30 = np.append(self.database['soil_Vs30'], self.database['soil_Vs30'], axis=0)
            magnitude = np.append(self.database['magnitude'], self.database['magnitude'], axis=0)
            rjb = np.append(self.database['Rjb'], self.database['Rjb'], axis=0)
            mechanism = np.append(self.database['mechanism'], self.database['mechanism'], axis=0)
            filename1 = np.append(self.database['Filename_1'], self.database['Filename_2'], axis=0)
            eq_id = np.append(self.database['EQID'], self.database['EQID'], axis=0)
            
            if self.database['Name'] == "NGA_W2":
                nga_num = np.append(self.database['NGA_num'], self.database['NGA_num'], axis=0)
            elif self.database['Name'] == "ESM_2018":
                station_code = np.append(self.database['station_code'], self.database['station_code'], axis=0)
            elif self.database['Name'] == "Global":
                station_code = np.append(self.database['station_code'], self.database['station_code'], axis=0)
                
        elif self.num_components == 2:
            if self.spectrum_definition == 'GeoMean':
                sa_known = np.sqrt(self.database['Sa_1'] * self.database['Sa_2'])
            elif self.spectrum_definition == 'SRSS':
                sa_known = np.sqrt(self.database['Sa_1'] ** 2 + self.database['Sa_2'] ** 2)
            elif self.spectrum_definition == 'ArithmeticMean':
                sa_known = (self.database['Sa_1'] + self.database['Sa_2']) / 2
            elif self.spectrum_definition == 'RotD50':
                sa_known = self.database['Sa_RotD50']
            elif self.spectrum_definition == 'RotD100':
                sa_known = self.database['Sa_RotD100']
            else:
                raise ValueError('Unexpected Sa definition, exiting...')
            
            vs30 = self.database['soil_Vs30']
            magnitude = self.database['magnitude']
            rjb = self.database['Rjb']
            mechanism = self.database['mechanism']
            filename1 = self.database['Filename_1']
            filename2 = self.database['Filename_2']
            eq_id = self.database['EQID']
            
            if self.database['Name'] == "NGA_W2":
                nga_num = self.database['NGA_num']
            elif self.database['Name'] == "ESM_2018":
                station_code = self.database['station_code']
            elif self.database['Name'] == "Global":
                station_code = self.database['station_code']

        else:
            raise ValueError('Selection can only be performed for one or two components at the moment, exiting...')
        
        # --- 2. Filtering Records (Only Sa <= 0 filter remains) ---
        
        # Sa cannot be negative or zero. Identify records to be excluded.
        not_allowed = np.unique(np.where(sa_known <= 0)[0]).tolist()
        
        # NOTE: Filtering by vs30_limits, mag_limits, rjb_limits, and mech_limits has been removed.
        
        # Finalize list of allowed indices
        not_allowed = (list(set(not_allowed)))
        allowed = [i for i in range(sa_known.shape[0])]
        for i in not_allowed:
            if i in allowed:
                allowed.remove(i)
        
        # --- 3. Apply Filter and Final Checks ---
        
        # Apply the filter to all data arrays
        sa_known = sa_known[allowed, :]
        vs30 = vs30[allowed]
        magnitude = magnitude[allowed]
        rjb = rjb[allowed]
        mechanism = mechanism[allowed]
        eq_id = eq_id[allowed]
        filename1 = filename1[allowed]
        
        # Handle filename2 filtering
        if self.num_components == 2:
            filename2 = filename2[allowed]
        else:
            filename2 = None

        # Handle database-specific metadata filtering
        if self.database['Name'] == "NGA_W2":
            if nga_num is not None: nga_num = nga_num[allowed]
            station_code = None
        elif self.database['Name'] == "ESM_2018":
            nga_num = None
            if station_code is not None: station_code = station_code[allowed]
        elif self.database['Name'] == "Global":
            nga_num = None
            if station_code is not None: station_code = station_code[allowed]
        
        # --- Period Matching (Ensure periods match the database exactly) ---
        
        record_periods = []
        
        # This loop attempts to find the index of every period in self.periods within the database periods.
        try:
            for period in self.periods:
                # Find the index of the period in the database periods array
                # [0][0] extracts the single index value from the result (np.array([index]))
                match_index = np.where(self.database['Periods'] == period)[0][0] 
                record_periods.append(match_index)
                
        except IndexError as e:
            # Catching the error that occurs when np.where returns an empty array, meaning a period is missing.
            raise ValueError(f"Target period {period} not found in database periods. All target periods must exist in the database.") from e
        
        # Final filter: select only the columns (periods) corresponding to the retained periods
        sample_big = sa_known[:, record_periods]
        
        # Check for invalid input (NaNs)
        if np.any(np.isnan(sample_big)):
            raise ValueError('NaNs found in input response spectra after period filtering.')
        
        # Check if enough records satisfy the criteria
        if self.num_records > len(eq_id):
            raise ValueError('There are not enough records which satisfy',
                            'the given record selection criteria...',
                            'Please use broaden your selection criteria...')
        
        # Return the filtered data
        return sample_big, vs30, magnitude, rjb, mechanism, filename1, filename2, nga_num, eq_id, station_code


    @staticmethod
    @njit
    def _find_rec_greedy(sample_small, scaling_factors, mu_ln, sigma_ln, rec_id, sample_big, error_weights, max_scale_factor, num_records, penalty):
        """
        Details
        -------
        Greedy subset modification algorithm
        The method is defined separately so that njit can be used as wrapper and the routine can be run faster

        Parameters
        ----------
        sample_small : numpy.ndarray (2-D)
            Spectra of the reduced candidate record set (num_records - 1)
        scaling_factors : numpy.ndarray (1-D)
            Scaling factors for all records in the filtered database
        mu_ln : numpy.ndarray (1-D)
            Logarthmic mean of the target spectrum (conditional or unconditional)
        sigma_ln : numpy.ndarray (1-D)
            Logarthmic standard deviation of the target spectrum (conditional or unconditional)
        rec_id : numpy.ndarray (1-D)
            Record IDs of the reduced candidate set records in the database (num_records - 1)
        sample_big : numpy.ndarray (2-D)
            Spectra of the records in the filtered database
        error_weights : numpy.ndarray (1-D) or list 
            Weights for error in mean, standard deviation and skewness
        max_scale_factor : float
            The maximum allowable scale factor
        num_records : int
            Number of ground motions to be selected.
        penalty : int
            > 0 to penalize selected spectra more than 3 sigma from the target at any period, 0 otherwise.

        Returns
        -------
        min_id : int
            ID of the new selected record with the scale factor closest to 1
        """
        def mean_numba(arr):
            """
            Computes the mean of a 2-D array along axis=0.
            Required for computations since njit is used as wrapper.
            """

            res = []
            for i in range(arr.shape[1]):
                res.append(arr[:, i].mean())

            return np.array(res)

        def std_numba(arr):
            """
            Computes the standard deviation of a 2-D array along axis=0.
            Required for computations since njit is used as wrapper.
            """

            res = []
            for i in range(arr.shape[1]):
                res.append(arr[:, i].std())

            return np.array(res)

        min_dev = 100000
        for j in range(sample_big.shape[0]):
            # Add to the sample the scaled spectrum
            temp = np.zeros((1, len(sample_big[j, :])))
            temp[:, :] = sample_big[j, :]
            sample_small_trial = np.concatenate((sample_small, temp + np.log(scaling_factors[j])), axis=0)
            dev_mean = mean_numba(sample_small_trial) - mu_ln  # Compute deviations from target
            dev_sig = std_numba(sample_small_trial) - sigma_ln
            dev_total = error_weights[0] * np.sum(dev_mean * dev_mean) + error_weights[1] * np.sum(dev_sig * dev_sig)

            # Check if we exceed the scaling limit
            if scaling_factors[j] > max_scale_factor or scaling_factors[j] < 1 / max_scale_factor or np.any(rec_id == j):
                dev_total = dev_total + 1000000
            # Penalize bad spectra
            elif penalty > 0:
                for m in range(num_records):
                    dev_total = dev_total + np.sum(np.abs(np.exp(sample_small_trial[m, :]) > np.exp(mu_ln + 3.0 * sigma_ln))) * penalty
                    dev_total = dev_total + np.sum(np.abs(np.exp(sample_small_trial[m, :]) < np.exp(mu_ln - 3.0 * sigma_ln))) * penalty

            # Should cause improvement and record should not be repeated
            if dev_total < min_dev:
                min_id = j
                min_dev = dev_total

        return min_id


    def _simulate_spectra(self):
        """
        Details
        -------
        Generates simulated response spectra with best matches to the target values.

        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
        """

        # Set initial seed for simulation
        if self.seed_value:
            np.random.seed(self.seed_value)
        else:
            np.random.seed(sum(gmtime()[:6]))

        dev_total_sim = np.zeros((self.num_simulations, 1))
        spectra = {}
        # Generate simulated response spectra with best matches to the target values
        for j in range(self.num_simulations):
            # It might be better to use the second function if cov_rank = np.linalg.matrix_rank(self.cov) < len(mu_ln)
            spectra[j] = np.exp(random_multivariate_normal(self.mu_ln, self.cov, self.num_records, 'LHS'))
            # specDict[j] = np.exp(np.random.multivariate_normal(self.mu_ln, self.cov, size=self.num_records))

            # how close is the mean of the spectra to the target
            dev_mean_sim = np.mean(np.log(spectra[j]), axis=0) - self.mu_ln
            # how close is the mean of the spectra to the target
            dev_sig_sim = np.std(np.log(spectra[j]), axis=0) - self.sigma_ln
            # how close is the skewness of the spectra to zero (i.e., the target)  
            dev_skew_sim = skew(np.log(spectra[j]), axis=0)
            # combine the three error metrics to compute a total error
            dev_total_sim[j] = self.error_weights[0] * np.sum(dev_mean_sim ** 2) + self.error_weights[1] * np.sum(dev_sig_sim ** 2) + 0.1 * (self.error_weights[2]) * np.sum(dev_skew_sim ** 2)

        recUse = np.argmin(np.abs(dev_total_sim))  # find the simulated spectra that best match the targets
        self.sim_spec = np.log(spectra[recUse])  # return the best set of simulations


    #----------------------------------------------------------------------------#
    #                                                                            #        
    #      Create Target Conditional Spectra Method Based on OQ Calculations     #
    #                                                                            #
    #----------------------------------------------------------------------------#
    def create_target(self, 
                      target_return_period, 
                      lon_lat, 
                      num_components=None, 
                      spectrum_definition='RotD50',
                      cs_results=None):
        """
        Generate the target spectrum for a SINGLE specified location (lon_lat) 
        and return period, aligning the period vector with the database periods.
        
        Parameters
        ----------
        target_return_period : int or str
            The return period (e.g., 475, 975, 2475).
        lon_lat : list[str]
            Two string values for longitude and latitude, e.g., ['12.523', '41.606'].
        spectrum_definition : str
            Type of target spectrum (e.g., 'RotD50', 'GeoMean').
        cs_results : dict, optional
            Preloaded conditional spectra results from OpenQuake. If None, uses 
            the results stored in self.target_spectra from process_oq_cs_calc().
        """
        
        # --- 1. Setup and Input Validation ---
        self.lon_lat = lon_lat 
        
        # Handle return period key
        if isinstance(target_return_period, (int, float)):
            target_rp_str = str(int(target_return_period))
            self.target_return_period = int(target_return_period)
        elif isinstance(target_return_period, str):
            target_rp_str = target_return_period
            self.target_return_period = target_return_period
        else:
            raise TypeError("target_return_period must be an integer, float, or string.")
        
        if len(lon_lat) != 2 or not all(isinstance(c, str) for c in lon_lat):
            raise TypeError("lon_lat must be a list of two strings, e.g., ['12.523', '41.606'].")
        
        target_loc_key = f"({lon_lat[0]}, {lon_lat[1]})"
        
        # Components and spectrum definition
        if num_components is None:
            num_components = {'Arbitrary': 1, 'GeoMean': 2, 'RotD50': 2, 'RotD100': 2}[spectrum_definition]
        
        self.num_components = num_components
        self.spectrum_definition = spectrum_definition
        
        # --- 2. Load Conditional Spectra Results ---
        cs_results_source = cs_results if cs_results is not None else self.target_spectra

        if not cs_results_source:
             raise RuntimeError("Conditional spectra results are not loaded. Please ensure you have run 'process_oq_cs_calc()' successfully before calling 'create_target' without passing the 'cs_results' argument.")
        
        # We need the conditioning IMT (e.g., SA(1.0)) to access the data. 
        # Assuming the first key in the location's dictionary is the IMT.
        try:
            loc_data = next(iter(cs_results_source[target_loc_key].values()))
        except KeyError:
            found_keys = list(cs_results_source.keys())
            raise ValueError(f"Location key '{target_loc_key}' corresponding to {lon_lat} "
                              f"was not found in OpenQuake output. Example keys: {found_keys[:2]}")
        except StopIteration:
            raise ValueError(f"No IMT data found for location {target_loc_key}.")

        
        # Robust return period key check (check for string or int keys)
        if target_rp_str in loc_data:
            return_period_key = target_rp_str
        elif target_rp_str.isdigit() and int(target_rp_str) in loc_data:
            return_period_key = int(target_rp_str)
        else:
            rp_keys_available = list(loc_data.keys())
            raise ValueError(f"Target return period '{target_rp_str}' not found for location {target_loc_key}. "
                             f"Available keys: {rp_keys_available}")
        
        # Extract OQ spectral data
        data = loc_data[return_period_key]
        oq_periods = np.array(data["period"])
        oq_mean = np.array(data["mean"])
        oq_mean_plus = np.array(data["mean+std"])
        oq_mean_minus = np.array(data["mean-std"])
        # This key is now guaranteed to exist if process_oq_cs_calc was run with the latest version
        oq_sigma_ln = np.array(data["sigma_ln"]) 
        
        # --- 3. Define Period Range from OQ and Align with Database ---
        # Find the indices in the database period vector that bound the OQ periods
        temp = np.abs(self.database["Periods"] - np.min(oq_periods))
        idx1 = np.where(temp == np.min(temp))[0][0]
        temp = np.abs(self.database["Periods"] - np.max(oq_periods))
        idx2 = np.where(temp == np.min(temp))[0][0]
        self.periods = self.database["Periods"][idx1:idx2 + 1]
        
        # --- 4. Interpolate the OQ spectrum onto these database-aligned periods ---
        interp_mean = interpolate.interp1d(oq_periods, oq_mean, kind='linear', fill_value="extrapolate")
        interp_sigma_ln = interpolate.interp1d(oq_periods, oq_sigma_ln, kind='linear', fill_value="extrapolate")

        mean = interp_mean(self.periods)
        sigma_ln = interp_sigma_ln(self.periods)
        
        # Interpolate the bounds for plotting verification only
        interp_mean_plus = interpolate.interp1d(oq_periods, oq_mean_plus, kind='linear', fill_value="extrapolate")
        interp_mean_minus = interpolate.interp1d(oq_periods, oq_mean_minus, kind='linear', fill_value="extrapolate")
        mean_plus = interp_mean_plus(self.periods)
        mean_minus = interp_mean_minus(self.periods)

        # --- 5. Compute Lognormal Parameters and Covariance ---
        mu_ln = np.log(mean)
        cov = np.diag(sigma_ln ** 2)
        
        # --- 6. Store Results ---
        self.periods_full = oq_periods
        self.mu_ln_full = np.log(oq_mean)
        self.sigma_ln_full = oq_sigma_ln
        self.cov_full = np.diag(self.sigma_ln_full ** 2)
        
        self.mu_ln = mu_ln
        self.sigma_ln = sigma_ln
        self.cov = cov
        self.mean_spectrum = mean
        self.std_spectrum = sigma_ln
                         
    #----------------------------------------------------------------------------#
    #                                                                            #        
    #               Select Records Based on Target Conditional Spectra           #
    #                                                                            #
    #----------------------------------------------------------------------------#
    def select_records(self, 
                      num_records=30, 
                      is_scaled=True, 
                      max_scale_factor=2.5, 
                      num_simulations=20,
                      seed_value=None, 
                      error_weights=[1, 2, 0.3], 
                      num_greedy_loops=2, 
                      penalty=0,
                      tolerance=10):
        """
        Perform ground motion selection assuming conditional spectra (CS).
        Always conditioned on the target intensity measure level at T* (self.Tstar).
        """
        
        # --- 1. Validation and Settings Setup ---
        if not hasattr(self, "periods_full"):
            raise AttributeError("Target spectrum attributes not found. Please call create_target() first.")
        
        # Settings
        self.num_records = num_records
        self.seed_value = seed_value
        self.error_weights = np.array(error_weights)
        self.max_scale_factor = max_scale_factor if max_scale_factor is not None else 10
        self.is_scaled = is_scaled
        self.num_greedy_loops = num_greedy_loops
        self.tolerance = tolerance
        self.penalty = penalty
        self.num_simulations = num_simulations
        
        # --- 2. Simulate Target Spectra ---
        self._simulate_spectra()  # Uses full target arrays
        
        # --- 3. Data Search and Period Alignment ---
        # NOTE: _search_database() now returns records already truncated to self.periods
        #sample_big, vs30, mag, rjb, mechanism, filename1, filename2, rsn, eq_id, station_code = self._search_database()
        sample_big, vs30, mag, rjb, mechanism, rsn, eq_id, station_code                       = self._search_global_flatfile()
        
        # --- 4. Compute IMLs at Conditioning Period(s) ---
        if not hasattr(self, "Tstar") or self.Tstar is None:
            raise AttributeError("Tstar must be defined before selecting records.")
            
        sample_big = np.log(sample_big)
        len_big = sample_big.shape[0]
        
        # --- Compute IMLs at conditioning period(s) Tstar ---
        # This interpolates the database records onto the Tstar period(s)
        f = interpolate.interp1d(self.periods, np.exp(sample_big), kind="linear", axis=1, fill_value="extrapolate")
        sample_big_imls = np.exp(np.sum(np.log(f(self.Tstar)), axis=1) / len(self.Tstar))
        
        # --- 5. Initialize Selection Containers ---
        rec_id = np.ones(self.num_records, dtype=int) * -1
        final_scale_factors = np.ones(self.num_records)
        sample_small = np.ones((self.num_records, sample_big.shape[1]))
        
        target_sim_spec = self.sim_spec
        
        # --- 6. Initial Record Selection ---
        
        # Find the mean spectral acceleration at Tstar from the target spectrum
        interp_target_mean = interpolate.interp1d(self.periods, self.mean_spectrum, kind='linear', fill_value="extrapolate")
        target_im_Tstar_scalar = interp_target_mean(self.Tstar).item()
        
        for i in range(self.num_records):
            error = np.zeros(len_big)
            scaling_factors = np.ones(len_big)
            
            if self.is_scaled:
                # Always scale to match IML at T*
                scaling_factors = target_im_Tstar_scalar / sample_big_imls
            
            mask = (1 / self.max_scale_factor < scaling_factors) & (scaling_factors < self.max_scale_factor)
            idxs = np.where(mask)[0]
            error[~mask] = 1e6 # Set error for records outside scale factor range to a high value
            
            # Scale the records that passed the check
            scaled_records = np.log(np.exp(sample_big[idxs, :]) * scaling_factors[mask].reshape(-1, 1))
            # Calculate the error for records that passed the check (using the index set 'idxs' to slice sample_big)
            error[mask] = np.sum((scaled_records - target_sim_spec[i, :]) ** 2, axis=1)
            
            # --- FIX: np.argmin(error) returns the correct global index into sample_big. ---
            min_id = np.argmin(error)
            
            # Assign the global index directly
            rec_id[i] = min_id
            
            if error.min() >= 1e6:
                # If the minimum error is still the dummy high value, no good match was found
                raise Warning(f"No good matches found for simulated spectrum {i+1}. Minimum error is high.")
            
            if self.is_scaled:
                # Use the global index to look up the scaling factor
                final_scale_factors[i] = scaling_factors[rec_id[i]]
            
            # Use the global index and the determined scale factor to store the resulting scaled spectrum
            sample_small[i, :] = np.log(np.exp(sample_big[rec_id[i], :]) * final_scale_factors[i])
        
        # --- 7. Greedy Refinement ---
        # Find indices where period does NOT equal Tstar (for error checking)
        ind2 = np.where(~np.isin(self.periods, self.Tstar))[0].tolist()
        
        for _ in range(self.num_greedy_loops):
            for i in range(self.num_records):
                sample_small_tmp = np.delete(sample_small, i, 0)
                rec_id_tmp = np.delete(rec_id, i)
                
                # Use the target IML at T* for scaling
                scaling_factors = target_im_Tstar_scalar / sample_big_imls
                
                min_id = self._find_rec_greedy(
                    sample_small_tmp, scaling_factors, self.mu_ln, self.sigma_ln,
                    rec_id_tmp, sample_big, self.error_weights,
                    self.max_scale_factor, self.num_records, self.penalty
                )
                
                final_scale_factors[i] = scaling_factors[min_id]
                new_scaled_sa = sample_big[min_id, :] + np.log(scaling_factors[min_id])
                
                sample_small[i, :] = new_scaled_sa
                rec_id[i] = min_id
            
            # Check convergence (exclude T*)
            median_error = np.max(np.abs(np.exp(np.mean(sample_small[:, ind2], axis=0)) - np.exp(self.mu_ln[ind2])) / np.exp(self.mu_ln[ind2])) * 100
            std_error = np.max(np.abs(np.std(sample_small[:, ind2], axis=0) - self.sigma_ln[ind2]) / self.sigma_ln[ind2]) * 100
            
            if median_error < self.tolerance and std_error < self.tolerance:
                break
        
        # --- 8. Final Outputs ---
        self.median_error = median_error
        self.std_error = std_error
        
        print("Ground motion selection (CS-based) completed.")
        print(f"For T \u2208 [{self.periods[0]:.2f} - {self.periods[-1]:.2f}]")
        print(f"Max error in median = {median_error:.2f} %")
        print(f"Max error in std. deviation = {std_error:.2f} %")
        
        rec_id = rec_id.tolist()
        self.rec_scale_factors = final_scale_factors
        self.rec_sa_ln = sample_small
        self.rec_vs30 = vs30[rec_id]
        self.rec_rjb = rjb[rec_id]
        self.rec_mag = mag[rec_id]
        self.rec_mech = mechanism[rec_id]
        self.rec_eq_id = eq_id[rec_id]
        #self.rec_file_h1 = filename1[rec_id]
        
        # if getattr(self, "num_components", 1) == 2:
        #     self.rec_file_h2 = filename2[rec_id]
        
        if self.database["Name"] == "NGA_W2":
            self.rec_rsn = rsn[rec_id]
        
        if self.database["Name"] == "ESM_2018":
            self.rec_station_code = station_code[rec_id]

    #----------------------------------------------------------------------------#
    #                                                                            #        
    #               Select Records Based on Target Conditional Spectra           #
    #                                                                            #
    #----------------------------------------------------------------------------#
    def plot_records(self, save=False, pflag=True):  
    
        """
        Details
        -------
        Plots the target spectrum (mean and uncertainty bounds) against the spectra
        of the selected records (individual records, mean, and dispersion bounds).
    
        Parameters
        ----------
        save : bool (optional)
            Flag to save the figure in PDF format (Default: False).
        pflag : bool (optional)
            Flag to show the figure interactively (Default: True).
    
        Returns
        -------
        None
        """
        
        if not hasattr(self, 'periods') or not hasattr(self, 'rec_sa_ln'):
            print("Error: Target periods or selected records data (self.rec_sa_ln) not available. Run select_records() first.")
            return
    
        plt.ioff()
    
        # --- Setup Ticks and Tstar Hatching ---
        # xticks and yticks to use for plotting
        xticks = [self.periods[0]]
        for x in [0.01, 0.1, 0.2, 0.5, 1, 5, 10]:
            if self.periods[0] < x < self.periods[-1]:
                xticks.append(x)
        xticks.append(self.periods[-1])
        # Ticks for Sa (Spectral Acceleration)
        yticks = [0.01, 0.1, 0.2, 0.5, 1, 2, 3, 5] 
    
        hatch = None
        if getattr(self, 'is_conditioned', 0) == 1 and hasattr(self, 'Tstar'):
            if isinstance(self.Tstar, (int, float)) or len(self.Tstar) == 1:
                Tstar_val = self.Tstar if isinstance(self.Tstar, (int, float)) else self.Tstar[0]
                hatch = [float(Tstar_val * 0.98), float(Tstar_val * 1.02)]
            elif isinstance(self.Tstar, np.ndarray):
                hatch = [float(self.Tstar.min()), float(self.Tstar.max())]
    
        # --- Create the Figure (Two Subplots: Mean/Records vs. Dispersion) ---
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        
        # Use target return period for title if available
        rp_str = getattr(self, 'target_return_period', 'Target')
        plt.suptitle(f'Target Spectrum ({rp_str} yrs) vs. Spectra of Selected Records', y=0.95)
    
        # --- AXIS 0: Response Spectra (Sa vs. Period) ---
        
        # 1. Plot Individual Selected Records (if requested)
        for i in range(self.num_records):
            # Plot individual records in gray
            ax[0].loglog(self.periods, np.exp(self.rec_sa_ln[i, :]), color='gray', 
                         lw=LINEWIDTH_1, label='Selected' if i == 0 else "_nolegend_")
                
        # 2. Plot Target Spectrum (Mean and Bounds)
        # Mean target spectrum
        ax[0].loglog(self.periods, np.exp(self.mu_ln), color='red', lw=LINEWIDTH_2, 
                     label=r'Target - $e^{\mu_{ln}}$')
        # Bounds (+/- 2*sigma) - Check for use_variance (optional)
        if getattr(self, 'use_variance', True): # Assuming use_variance default is True if not set
            ax[0].loglog(self.periods, np.exp(self.mu_ln + 2 * self.sigma_ln), color='red', linestyle='--', lw=LINEWIDTH_2,
                         label=r'Target - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
            ax[0].loglog(self.periods, np.exp(self.mu_ln - 2 * self.sigma_ln), color='red', linestyle='--', lw=LINEWIDTH_2,
                         label="_nolegend_")
    
        # 3. Plot Selected Records Ensemble (Mean and Bounds)
        # Mean selected spectrum
        ax[0].loglog(self.periods, np.exp(np.mean(self.rec_sa_ln, axis=0)), color='blue', lw=LINEWIDTH_2,
                     label=r'Selected - $e^{\mu_{ln}}$')
        # Bounds (+/- 2*std)
        ax[0].loglog(self.periods, np.exp(np.mean(self.rec_sa_ln, axis=0) + 2 * np.std(self.rec_sa_ln, axis=0)),
                     color='blue', linestyle='--', lw=LINEWIDTH_2, label=r'Selected - $e^{\mu_{ln}\mp 2\sigma_{ln}}$')
        ax[0].loglog(self.periods, np.exp(np.mean(self.rec_sa_ln, axis=0) - 2 * np.std(self.rec_sa_ln, axis=0)),
                     color='blue', linestyle='--', lw=LINEWIDTH_2, label="_nolegend_")
    
        # Axis 0 Formatting
        ax[0].set_xlim([self.periods[0], self.periods[-1]])
        ax[0].set_xticks(xticks)
        ax[0].get_xaxis().set_major_formatter(ScalarFormatter())
        ax[0].get_xaxis().set_minor_formatter(NullFormatter())
        ax[0].set_yticks(yticks)
        ax[0].get_yaxis().set_major_formatter(ScalarFormatter())
        ax[0].get_yaxis().set_minor_formatter(NullFormatter())
        ax[0].set_xlabel('Period [sec]')
        ax[0].set_ylabel('Spectral Acceleration [g]')
        ax[0].grid(True, which='major', linestyle='-')
        ax[0].grid(True, which='minor', linestyle='--', alpha=0.5)
    
        # Legend for Axis 0 (Handle duplicate labels)
        handles, labels = ax[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[0].legend(by_label.values(), by_label.keys(), frameon=False, loc='best')
    
        # Hatch for Tstar (Conditional Spectrum)
        if hatch:
            ax[0].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)
    
        # --- AXIS 1: Dispersion (Sigma vs. Period) ---
    
        # Plot Target Dispersion
        ax[1].semilogx(self.periods, self.sigma_ln, color='red', linestyle='--', lw=LINEWIDTH_2, 
                       label=r'Target - $\sigma_{ln}$')
    
        # Plot Selected Records Dispersion
        ax[1].semilogx(self.periods, np.std(self.rec_sa_ln, axis=0), color='black', linestyle='--', lw=LINEWIDTH_2,
                       label=r'Selected - $\sigma_{ln}$')
            
        # Axis 1 Formatting
        ax[1].set_xlabel('Period [sec]')
        ax[1].set_ylabel('Dispersion ($\ln SA$)')
        ax[1].grid(True, which='major', linestyle='-')
        ax[1].grid(True, which='minor', linestyle='--', alpha=0.5)
        ax[1].legend(frameon=False, loc='best')
        ax[1].set_xlim([self.periods[0], self.periods[-1]])
        ax[1].set_xticks(xticks)
        ax[1].get_xaxis().set_major_formatter(ScalarFormatter())
        ax[1].get_xaxis().set_minor_formatter(NullFormatter())
        ax[1].set_ylim(bottom=0)
        ax[1].get_yaxis().set_major_formatter(ScalarFormatter())
        ax[1].get_yaxis().set_minor_formatter(NullFormatter())
        
        # Hatch for Tstar (Conditional Spectrum)
        if hatch:
            ax[1].axvspan(hatch[0], hatch[1], facecolor='red', alpha=0.3)
            
        # --- Add error annotations to subplots if available ---
        if hasattr(self, "median_error") and hasattr(self, "std_error"):
            # Left subplot: median error
            ax[0].text(0.05, 0.95,
                       f"Maximum Error in Median = {self.median_error:.2f}%",
                       transform=ax[0].transAxes,
                       fontsize=11,
                       color='black',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
            # Right subplot: std. deviation error
            ax[1].text(0.05, 0.95,
                       f"Maximum Error in $\sigma$ = {self.std_error:.2f}%",
                       transform=ax[1].transAxes,
                       fontsize=11,
                       color='black',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # --- Final Actions ---
        if save:
            try:
                # Compose full path (inside figure_path)
                fname_plot = self.figure_path / f'Target_vs_Selected_Records_{rp_str}.png'
                # Save figure
                plt.savefig(fname_plot, format='png', dpi=300)
        
            except Exception as e:
                print(f"Warning: Could not save figure. Error: {e}")

    #----------------------------------------------------------------------------#
    #                                                                            #        
    #                        Write Selected Records to TXT                       #
    #                                                                            #
    #----------------------------------------------------------------------------#
    def write_records(self, record_type='acc', zip_parent_path=''):
        """
        Details
        -------
        Writes the object as pickle, selected and scaled records as .txt files.

        Parameters
        ----------
        record_type : str, optional
            option to choose the type of time history to be written.
            'acc' : for the acceleration series, units: g
            'vel' : for the velocity series, units: g * sec
            'disp': for the displacement series: units: g * sec2
        zip_parent_path : str, optional
            This is option could be used if the user already has all the
            records in database. This is the folder path which contains
            "database.zip" file (e.g., database could be NGA_W2 or ESM_2018). 
            The records must be placed inside zip_parent_path/database.zip/database/
            The default is ''.

        Notes
        -----
        0: no, 1: yes

        Returns
        -------
        None.
        """

        def save_signal(path, unscaled_acc, sf, dt):
            """
            Details
            -------
            Saves the final signal to the specified path.

            Parameters
            ----------
            path : str
                path of the file to save
            unscaled_acc : numpy.ndarray
                unscaled acceleration series
            sf : float
                scaling factor
            dt : float
                time step 

            Returns
            -------
            None.
            """

            if record_type == 'vel':  # integrate once if velocity
                signal = integrate.cumtrapz(unscaled_acc * sf, dx=dt, initial=0)

            elif record_type == 'disp':  # integrate twice if displacement
                signal = integrate.cumtrapz(integrate.cumtrapz(unscaled_acc * sf, dx=dt, initial=0), dx=dt, initial=0)

            else:
                signal = unscaled_acc * sf

            np.savetxt(path, signal, fmt='%1.5e')

        # Create the ground-motion records path            
        self.gmrs_path     = self.export_path / 'gmrs' / self.target_return_period      # Output path for ground motion records
        self.gmrs_path.mkdir(parents=True, exist_ok=True)
        
        # set the directories and file names
        try:  # this will work if records are downloaded
            zip_name = self.unscaled_rec_file
        except AttributeError:
            zip_name = os.path.join(zip_parent_path, self.database['Name'] + '.zip')
        size = len(self.rec_file_h1)
        dts = np.zeros(size)
        path_h1 = os.path.join(self.gmrs_path, 'GMR_names.txt')
        if self.num_components == 2:
            path_h1 = os.path.join(self.grms_path, 'GMR_H1_names.txt')
            path_h2 = os.path.join(self.gmrs_path, 'GMR_H2_names.txt')
            h2s = open(path_h2, 'w')
        h1s = open(path_h1, 'w')

        # Get record paths for # NGA_W2 or ESM_2018
        if zip_name != os.path.join(zip_parent_path, self.database['Name'] + '.zip'):
            rec_paths1 = self.rec_file_h1
            if self.num_components == 2:
                rec_paths2 = self.rec_file_h2
        else:
            rec_paths1 = [self.database['Name'] + '/' + self.rec_file_h1[i] for i in range(size)]
            if self.num_components == 2:
                rec_paths2 = [self.database['Name'] + '/' + self.rec_file_h2[i] for i in range(size)]

        # Read contents from zipfile
        contents1 = content_from_zip(rec_paths1, zip_name)  # H1 gm components
        if self.num_components == 2:
            contents2 = content_from_zip(rec_paths2, zip_name)  # H2 gm components

        # Start saving records
        for i in range(size):

            # Read the record files
            if self.database['Name'].startswith('NGA'):  # NGA
                dts[i], npts1, _, _, inp_acc1 = read_nga(in_filename=self.rec_file_h1[i], content=contents1[i])
                gmr_file1 = self.rec_file_h1[i].replace('/', '_')[:-4] + '_' + record_type.upper() + '.txt'

                if self.num_components == 2:  # H2 component
                    _, npts2, _, _, inp_acc2 = read_nga(in_filename=self.rec_file_h2[i], content=contents2[i])
                    gmr_file2 = self.rec_file_h2[i].replace('/', '_')[:-4] + '_' + record_type.upper() + '.txt'

            elif self.database['Name'].startswith('ESM'):  # ESM
                dts[i], npts1, _, _, inp_acc1 = read_esm(in_filename=self.rec_file_h1[i], content=contents1[i])
                gmr_file1 = self.rec_file_h1[i].replace('/', '_')[:-11] + '_' + record_type.upper() + '.txt'
                if self.num_components == 2:  # H2 component
                    _, npts2, _, _, inp_acc2 = read_esm(in_filename=self.rec_file_h2[i], content=contents2[i])
                    gmr_file2 = self.rec_file_h2[i].replace('/', '_')[:-11] + '_' + record_type.upper() + '.txt'

            # Write the record files
            if self.num_components == 2:
                # ensure that two acceleration signals have the same length, if not add zeros.
                npts = max(npts1, npts2)
                temp1 = np.zeros(npts)
                temp1[:npts1] = inp_acc1
                inp_acc1 = temp1.copy()
                temp2 = np.zeros(npts)
                temp2[:npts2] = inp_acc2
                inp_acc2 = temp2.copy()

                # H2 component
                save_signal(os.path.join(self.gmrs_path, gmr_file2), inp_acc2, self.rec_scale_factors[i], dts[i])
                h2s.write(gmr_file2 + '\n')

            # H1 component
            save_signal(os.path.join(self.gmrs_path, gmr_file1), inp_acc1, self.rec_scale_factors[i], dts[i])
            h1s.write(gmr_file1 + '\n')

        # Time steps
        np.savetxt(os.path.join(self.gmrs_path, 'GMR_dts.txt'), dts, fmt='%.5f')
        # Scale factors
        np.savetxt(os.path.join(self.gmrs_path, 'GMR_sf_used.txt'), np.array([self.rec_scale_factors]).T, fmt='%1.5f')
        # Close the files
        h1s.close()
        if self.num_components == 2:
            h2s.close()

        print(f"Finished writing process, the files are located in\n{self.export_path}")

    #----------------------------------------------------------------------------#
    #                                                                            #        
    #                           Download Selected Records                        #
    #                                                                            #
    #----------------------------------------------------------------------------#
    def download(self, username=None, password=None, token_path=None, sleeptime=2, browser='chrome'):
        """
        Details
        -------
        This function has been created as a web automation tool in order to
        download unscaled record time histories from either
        NGA-West2 (https://ngawest2.berkeley.edu/) or ESM databases (https://esm-db.eu/).

        Notes
        -----
        Either of google-chrome or mozilla-firefox should have been installed priorly to download from NGA-West2.

        Parameters
        ----------
        username : str
            Account username (e-mail),  e.g. 'example_username@email.com'.
        password : str
            Account password, e.g. 'example_password123456'.
        sleeptime : int, optional
            Time (sec) spent between each browser operation. This can be increased or decreased depending on the internet speed.
            Used in the case of database='NGA_W2'
            The default is 2
        browser : str, optional
            The browser to use for download purposes. Valid entries are: 'chrome' or 'firefox'. 
            Used in the case of database='NGA_W2'
            The default is 'chrome'.

        Returns
        -------
        None
        """
        
        if self.database['Name'] == 'ESM_2018':

            if token_path is None:
                # In order to access token file must be retrieved initially.
                # copy paste the readily available token.txt into EzGM or generate new one using get_esm_token method.
                if username is None or password is None:
                    raise ValueError('You have to enter either credentials or path to the token to download records from ESM database')
                else:
                    get_esm_token(username, password)
                    token_path = 'token.txt'

            self._esm2018_download(token_path)

        elif self.database['Name'] == 'NGA_W2':

            if username is None or password is None:
                raise ValueError('You have to enter either credentials  to download records from NGA-West2 database')

            self._ngaw2_download(username, password, sleeptime, browser)

        else:
            raise NotImplementedError('You have to use either of ESM_2018 or NGA_W2 databases to use download method.')

    def _esm2018_download(self, token_path=None):
        """

        Details
        -------
        This function has been created as a web automation tool in order to
        download unscaled record time histories from ESM database
        (https://esm-db.eu/) based on their event ID and station_codes.

        Parameters
        ----------
        username : str
            Account username (e-mail),  e.g. 'example_username@email.com'.
        password : str
            Account password, e.g. 'example_password123456'.
        

        Returns
        -------
        None.

        """

        print('\nStarted executing download method to retrieve selected records from https://esm-db.eu')

        # temporary zipfile name
        zip_temp = os.path.join(self.export_path, 'output_temp.zip')
        # temporary folder to extract files
        folder_temp = os.path.join(self.export_path, 'output_temp')

        for i in range(self.num_records):
            print('Downloading %d/%d...' % (i + 1, self.num_records))
            event = self.rec_eq_id[i]
            station = self.rec_station_code[i]
            params = (
                ('eventid', event),
                ('data-type', 'ACC'),
                ('station', station),
                ('format', 'ascii'),
            )
            files = {
                'message': ('path/to/token.txt', open(token_path, 'rb')),
            }

            url = 'https://esm-db.eu/esmws/eventdata/1/query'

            req = requests.post(url=url, params=params, files=files)

            if req.status_code == 200:
                with open(zip_temp, "wb") as zf:
                    zf.write(req.content)

                with zipfile.ZipFile(zip_temp, 'r') as zipObj:
                    zipObj.extractall(folder_temp)
                os.remove(zip_temp)

            else:
                if req.status_code == 403:
                    sys.exit('Problem with ESM download. Maybe the token is no longer valid')
                else:
                    sys.exit('Problem with ESM download. Status code: ' + str(req.status_code))

        # create the output zipfile for all downloaded records
        time_tag = gmtime()
        time_tag_str = f'{time_tag[0]}'
        for i in range(1, len(time_tag)):
            time_tag_str += f'_{time_tag[i]}'
        file_name = os.path.join(self.export_path, f'unscaled_records_{time_tag_str}.zip')
        with zipfile.ZipFile(file_name, 'w', zipfile.ZIP_DEFLATED) as zipObj:
            len_dir_path = len(folder_temp)
            for root, _, files in os.walk(folder_temp):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipObj.write(file_path, file_path[len_dir_path:])

        shutil.rmtree(folder_temp)
        self.unscaled_rec_file = file_name
        print(f'Downloaded files are located in\n{self.unscaled_rec_file}')

    def _ngaw2_download(self, username, password, sleeptime, browser):
        """
        Details
        -------
        This function has been created as a web automation tool in order to
        download unscaled record time histories from NGA-West2 Database
        (https://ngawest2.berkeley.edu/) by Record Sequence Numbers (RSNs).

        Notes
        -----
        Either of google-chrome or mozilla-firefox should have been installed priorly.

        Parameters
        ----------
        username : str
            Account username (e-mail),  e.g. 'example_username@email.com'.
        password : str
            Account password, e.g. 'example_password123456'.
        sleeptime : int, default is 3
            Time (sec) spent between each browser operation. This can be increased or decreased depending on the internet speed.
        browser : str, default is 'chrome'
            The browser to use for download purposes. Valid entries are: 'chrome' or 'firefox'.

        Returns
        -------
        None
        """

        def dir_size(download_dir):
            """
            Details
            -------
            Measures download directory size

            Parameters
            ----------
            download_dir : str
                Directory for the output time histories to be downloaded

            Returns
            -------
            total_size : float
                Measured size of the download directory

            """

            total_size = 0
            for path, dirs, files in os.walk(download_dir):
                for f in files:
                    fp = os.path.join(path, f)
                    total_size += os.path.getsize(fp)
            return total_size

        def download_wait(download_dir):
            """
            Details
            -------
            Waits for download to finish, and an additional amount of time based on the predefined sleeptime variable.

            Parameters
            ----------
            download_dir : str
                Directory for the output time histories to be downloaded.

            Returns
            -------
            None
            """
            delta_size = 100
            flag = 0
            flag_lim = 5
            while delta_size > 0 and flag < flag_lim:
                size_0 = dir_size(download_dir)
                sleep(1.5 * sleeptime)
                size_1 = dir_size(download_dir)
                if size_1 - size_0 > 0:
                    delta_size = size_1 - size_0
                else:
                    flag += 1
                    print('Finishing in', flag_lim - flag, '...')

        def set_driver(download_dir, browser):
            """
            Details
            -------
            This function starts the webdriver in headless mode.

            Parameters
            ----------
            download_dir : str
                Directory for the output time histories to be downloaded.
            browser : str, default is 'chrome'
                The browser to use for download purposes. Valid entries are: 'chrome' or 'firefox'

            Returns
            -------
            driver : selenium webdriver object
                Driver object used to download NGA_W2 records.
            """

            print('Getting the webdriver to use...')

            # Check if ipython is installed
            try:
                __IPYTHON__
                _in_ipython_session = True
            except NameError:
                _in_ipython_session = False

            try:
                # Running on Google Colab
                if _in_ipython_session and 'google.colab' in str(get_ipython()):
                    os.system('apt-get update')
                    os.system('sudo apt install chromium-chromedriver')
                    os.system('sudo cp /usr/lib/chromium-browser/chromedriver /usr/bin')
                    options = webdriver.ChromeOptions()
                    options.add_argument('-headless')
                    options.add_argument('-no-sandbox')
                    options.add_argument('-disable-dev-shm-usage')
                    prefs = {"download.default_directory": download_dir}
                    options.add_experimental_option("prefs", prefs)
                    driver = webdriver.Chrome('chromedriver', options=options)

                # Running on Binder or Running on personal computer (PC) using firefox
                elif (_in_ipython_session and 'jovyan' in os.getcwd()) or browser == 'firefox':
                    gdd = GeckoDriverDownloader()
                    driver_path = gdd.download_and_install()
                    options = webdriver.firefox.options.Options()
                    options.headless = True
                    options.set_preference("browser.download.folderList", 2)
                    options.set_preference("browser.download.dir", download_dir)
                    options.set_preference('browser.download.useDownloadDir', True)
                    options.set_preference('browser.helperApps.neverAsk.saveToDisk', 'application/zip')
                    driver = webdriver.Firefox(executable_path=driver_path[1], options=options)

                # Running on personal computer (PC) using chrome
                elif browser == 'chrome':
                    gdd = ChromeDriverDownloader()
                    driver_path = gdd.download_and_install(version = 'compatible')
                    options = webdriver.ChromeOptions()
                    prefs = {"download.default_directory": download_dir}
                    options.add_experimental_option("prefs", prefs)
                    options.headless = True
                    driver = webdriver.Chrome(executable_path=driver_path[1], options=options)

                print('Webdriver is obtained successfully.')

                return driver

            except RuntimeError:
                print('Failed to get webdriver.')
                raise

        def sign_in(driver, username, password):
            """

            Details
            -------
            This function signs in to 'https://ngawest2.berkeley.edu/' with
            given account credentials.

            Parameters
            ----------
            driver : selenium webdriver object
                Driver object used to download NGA_W2 records.
            username : str
                Account username (e-mail), e.g.: 'username@mail.com'.
            password : str
                Account password, e.g.: 'password!12345'.

            Returns
            -------
            driver : selenium webdriver object
                Driver object used to download NGA_W2 records.

            """
            # TODO: For Selenium >= 4.3.0
            #  Deprecated find_element_by_* and find_elements_by_* are now removed (#10712)
            #  https://stackoverflow.com/questions/72773206/selenium-python-attributeerror-webdriver-object-has-no-attribute-find-el
            #  Modify the ngaw2_download method to use greater versions of Selenium than 4.2.0
            print("Signing in with credentials...")
            driver.get('https://ngawest2.berkeley.edu/users/sign_in')
            driver.find_element_by_id('user_email').send_keys(username)
            driver.find_element_by_id('user_password').send_keys(password)
            driver.find_element_by_id('user_submit').click()

            try:
                alert = driver.find_element_by_css_selector('p.alert')
                warn = alert.text
            except BaseException as e:
                warn = None
                print(e)

            if str(warn) == 'Invalid email or password.':
                driver.quit()
                raise Warning('Invalid email or password.')
            else:
                print('Signed in successfully.')

            return driver

        def download(rsn, download_dir, driver):
            """

            Details
            -------
            This function dowloads the timehistories which have been indicated with their record sequence numbers (rsn)
            from 'https://ngawest2.berkeley.edu/'.

            Parameters
            ----------
            rsn : str
                A string variable contains RSNs to be downloaded which uses ',' as delimiter
                between RNSs, e.g.: '1,5,91,35,468'.
            download_dir : str
                Directory for the output time histories to be downloaded.
            driver : class object, (selenium webdriver)
                Driver object used to download NGA_W2 records.

            Returns
            -------
            None

            """
            print("Listing the Records...")
            driver.get('https://ngawest2.berkeley.edu/spectras/new?sourceDb_flag=1')
            sleep(sleeptime)
            driver.find_element_by_xpath("//button[@type='button']").submit()
            sleep(sleeptime)
            driver.find_element_by_id('search_search_nga_number').send_keys(rsn)
            sleep(sleeptime)
            driver.find_element_by_xpath(
                "//button[@type='button' and @onclick='uncheck_plot_selected();reset_selectedResult();OnSubmit();']").submit()
            sleep(1.5 * sleeptime)
            try:
                note = driver.find_element_by_id('notice').text
                print(note)
            except BaseException as e:
                note = 'NO'
                error = e

            if 'NO' in note:
                driver.set_window_size(800, 800)
                driver.save_screenshot(os.path.join(self.export_path, 'download_error.png'))
                driver.quit()
                raise Warning("Could not be able to download records!"
                              "Either they no longer exist in database"
                              "or you have exceeded the download limit")
            else:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
                sleep(sleeptime)
                driver.find_element_by_xpath("//button[@type='button' and @onclick='getSelectedResult(true)']").click()
                obj = driver.switch_to.alert
                msg = obj.text
                print(msg)
                sleep(sleeptime)
                obj.accept()
                sleep(sleeptime)
                obj = driver.switch_to.alert
                msg = obj.text
                print(msg)
                sleep(sleeptime)
                obj.accept()
                sleep(sleeptime)
                download_wait(download_dir)
                driver.quit()

        print('\nStarted executing download method to retrieve selected records from https://ngawest2.berkeley.edu')

        self.username = username
        self.pwd = password
        driver = set_driver(self.export_path, browser)
        driver = sign_in(driver, self.username, self.pwd)
        rsn = ''
        for i in self.rec_rsn:
            rsn += str(int(i)) + ','
        rsn = rsn[:-1:]
        files_before_download = set(os.listdir(self.export_path))
        download(rsn, self.export_path, driver)
        files_after_download = set(os.listdir(self.export_path))
        downloaded_file = str(list(files_after_download.difference(files_before_download))[0])
        file_extension = downloaded_file[downloaded_file.find('.')::]
        time_tag = gmtime()
        time_tag_str = f'{time_tag[0]}'
        for i in range(1, len(time_tag)):
            time_tag_str += f'_{time_tag[i]}'
        new_file_name = f'unscaled_records_{time_tag_str}{file_extension}'
        downloaded_file = os.path.join(self.export_path, downloaded_file)
        downloaded_file_rename = os.path.join(self.export_path, new_file_name)
        os.rename(downloaded_file, downloaded_file_rename)
        self.unscaled_rec_file = downloaded_file_rename
        print(f'Downloaded files are located in\n{self.unscaled_rec_file}')

