import math
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import genpareto, lognorm
from typing import Optional, Dict, Union, List
from pydantic import BaseModel, Field, validator

warnings.filterwarnings('ignore')

## Model and sub-components definition and validation classes
class component_data_model(BaseModel):
    
    """
    Represents a component's data, including its ID, description, EDP (Engineering Demand Parameter), 
    typology, performance group, quantity, and number of damage states.
    
    Attributes
    ----------
    Component_ID : Optional[int]
        Unique identifier for the component. Should be a positive integer.
    
    Description : Optional[str]
        Optional textual description of the component.
    
    EDP : str
        The Engineering Demand Parameter (e.g., Peak Storey Drift, Peak Floor Acceleration).
    
    Typology : str
        The type of component, such as structural or non-structural.
    
    Performance_Group : Optional[int]
        Optional group number for classification purposes.
    
    Quantity : float
        The quantity of the component present in the system.
    
    Damage_States : int
        The number of defined damage states for the component.
    
    Validators
    ----------
    Component_ID : Ensures Component_ID is a positive integer.
    Performance_Group, Component_ID : Allows these fields to be `None` or `NaN` if needed.
    """
    
    Component_ID:      Optional[int] = Field(alias="Component ID")
    Description:       Optional[str] = None
    EDP:                        str
    Typology:                   str
    Performance_Group: Optional[int] = Field(alias="Performance Group")
    Quantity:                  float
    Damage_States:               int = Field(alias="Damage States")

    @validator('Component_ID')
    def validate_id(cls, v):
        if v is not None and v < 0:
            raise ValueError('Component ID must be a positive integer')
        return v

    @validator('Performance_Group', 'Component_ID', pre=True)
    def allow_none(cls, v):
        if v is None or np.isnan(v):
            return None
        else:
            return v

class correlation_tree_model(BaseModel):
    """
    Represents a model for a correlation tree, which tracks dependencies between components.
    
    Attributes
    ----------
    ID : int
        The unique identifier for the correlation entry. Must be a positive integer.
    
    dependent_on_item : str
        The name of the item that the component is dependent on.
    
    Validators
    ----------
    ID : Ensures ID is a positive integer.
    """
    
    ID: int
    dependent_on_item: str = Field(alias="DEPENDENT ON ITEM")
    @validator('ID')
    def validate_id(cls, vid):
        if vid < 0:
            raise ValueError('Component ID must be a positive integer')
        return vid

class item_base(BaseModel):
    """
    Represents the base structure for items in a model, storing a dictionary of arrays for each item.
    
    Attributes
    ----------
    RootModel : Dict[str, np.ndarray]
        A dictionary where the keys are strings and the values are numpy arrays.
    
    Config
    ------
    Allows arbitrary types in the model.
    """
    
    RootModel: Dict[str, np.ndarray]
    class Config:
        arbitrary_types_allowed = True

class items_model(BaseModel):
    """
    Represents a collection of items, organized by a dictionary where the keys are integers and 
    the values are instances of `item_base`.

    Attributes
    ----------
    RootModel : Dict[int, item_base]
        A dictionary where each key represents a unique item ID and the value is an instance of `item_base`.
    
    Config
    ------
    Allows arbitrary types in the model.
    """
    RootModel: Dict[int, item_base]

class fragility_model(BaseModel):
    """
    Represents a fragility model, which includes the Engineering Demand Parameter (EDP) 
    and a collection of items associated with the fragility data.

    Attributes
    ----------
    EDP : np.ndarray
        An array representing the Engineering Demand Parameter (e.g., Peak Storey Drift, Peak Floor Acceleration).
    
    ITEMs : items_model
        A model representing the items in the fragility function.

    Config
    ------
    Allows arbitrary types in the model.
    """
    EDP: np.ndarray
    ITEMs: items_model
    class Config:
        arbitrary_types_allowed = True

class ds_model(BaseModel):
    """
    Represents the damage state model, which stores a dictionary of damage states for each item in the system.

    Attributes
    ----------
    RootModel : Dict[int, Dict[int, np.ndarray]]
        A dictionary where the first level key represents the item ID and the second level key 
        represents the damage state ID. The values are numpy arrays representing damage states.

    Config
    ------
    Allows arbitrary types in the model.
    """
    RootModel: Dict[int, Dict[int, np.ndarray]]
    class Config:
        arbitrary_types_allowed = True

class cost_model(BaseModel):
    """
    Represents the cost model for components, where the costs are organized in a dictionary.

    Attributes
    ----------
    RootModel : Dict[int, np.ndarray]
        A dictionary where the keys represent component IDs and the values are numpy arrays representing costs.

    Config
    ------
    Allows arbitrary types in the model.
    """
    RootModel: Dict[int, np.ndarray]
    class Config:
        arbitrary_types_allowed = True

class simulation_model(BaseModel):
    """
    Represents the simulation model for costs, which associates costs with specific components.

    Attributes
    ----------
    RootModel : Dict[int, cost_model]
        A dictionary where the keys represent component IDs and the values are instances of `cost_model`.

    Config
    ------
    Allows arbitrary types in the model.
    """
    RootModel: Dict[int, cost_model]
    class Config:
        arbitrary_types_allowed = True

class fitting_model(BaseModel):
    """
    Represents a fitting model, which includes the fitted parameters and their covariance matrix.

    Attributes
    ----------
    popt : np.ndarray
        The optimized fitting parameters.
    
    pcov : np.ndarray
        The covariance matrix of the fitted parameters.

    Config
    ------
    Allows arbitrary types in the model.
    """
    popt: np.ndarray
    pcov: np.ndarray
    class Config:
        arbitrary_types_allowed = True

class fitting_parameters_model(BaseModel):
    """
    Represents the fitting parameters for multiple models, organized by a dictionary of fitting models.

    Attributes
    ----------
    RootModel : Dict[str, fitting_model]
        A dictionary where the keys are fitting model names (strings) and the values are instances of `fitting_model`.
    """
    RootModel: Dict[str,fitting_model]

class fitted_loss_model(BaseModel):
    """
    Represents the fitted loss functions, which store the loss values for each model.

    Attributes
    ----------
    RootModel : Dict[str, np.ndarray]
        A dictionary where the keys are model names (strings) and the values are numpy arrays representing the fitted loss values.

    Config
    ------
    Allows arbitrary types in the model.
    """    
    RootModel: Dict[str, np.ndarray]
    class Config:
        arbitrary_types_allowed = True

class loss_model(BaseModel):
    """
    Represents the loss model, which includes both the total loss and the loss ratio for each component.

    Attributes
    ----------
    loss : Dict[int, Dict[Union[int, str], float]]
        A dictionary where the first level key represents component IDs, the second level key represents 
        damage states or performance levels, and the values are the corresponding loss values.
    
    loss_ratio : Dict[int, Dict[Union[int, str], float]]
        A dictionary where the first level key represents component IDs, the second level key represents 
        damage states or performance levels, and the values are the corresponding loss ratios.
    """
    loss: Dict[int, Dict[Union[int, str], float]]
    loss_ratio: Dict[int, Dict[Union[int, str], float]]

class slf_model(BaseModel):
    """
    Represents the Storey Loss Function (SLF) for a specific component, including directionality, 
    component type, storey levels, Engineering Demand Parameter (EDP), and the SLF values.

    Attributes
    ----------
    directionality : Optional[int]
        The directionality of the analysis, if applicable.
    
    component_type : str
        The type of the component (e.g., structural, non-structural).
    
    storey : Optional[Union[int, List[int]]]
        The storey level(s) associated with the SLF.
    
    edp : str
        The Engineering Demand Parameter (EDP) for the SLF (e.g., Peak Storey Drift).
    
    edp_range : List[float]
        The range of EDP values.
    
    slf : List[float]
        The calculated Storey Loss Function values.
    """    
    directionality: Optional[int] = Field(alias="Directionality")
    component_type: str = Field(alias="Component-type")
    storey: Optional[Union[int, List[int]]] = Field(aliast="Storey")
    edp: str
    edp_range: List[float]
    slf: List[float]

class slf_generator:
    """
    Storey-Loss-Function (SLF) Generator for Storey-Based Loss Assessment.

    This class automates the generation of Storey Loss Functions (SLFs) using fragility, consequence, and quantity data.
    It applies a probabilistic approach to quantify the loss and its distribution across various storeys of a building under seismic loading. 

    The tool uses input data about the building components, engineering demand parameters (EDPs), and regression models to generate SLFs that can be used for seismic risk analysis. The SLFs can then be employed in loss estimation frameworks like FEMA P-58 or similar approaches.

    -----------------
    References:
    -----------------
    1) Ramirez and Miranda (2009) "BUILDING-SPECIFIC LOSS ESTIMATION METHODS & TOOLS FOR SIMPLIFIED PERFORMANCE-BASED EARTHQUAKE ENGINEERING," John A. Blume Earthquake Engineering Center, Department of Civil and Environmental Engineering, Stanford University.
        
    2) Shahnazaryan D, O'Reilly GJ, Monteiro R. "Story loss functions for seismic design and assessment: Development of tools and application," Earthquake Spectra 2021; 37(4): 2813–2839. DOI: 10.1177/87552930211023523.
    
    3) Shahnazaryan D, O'Reilly GJ, Monteiro R. "Development of a Python-Based Storey Loss Function Generator," COMPDYN 2021 - 8th International Conference on Computational Methods in Structural Dynamics and Earthquake Engineering, 2021. DOI: 10.7712/120121.8659.18567.

    -----------------
    Acknowledgements:
    -----------------
    This code is based on the original work developed by Dr. Davit Shahnazaryan, available at 
    https://github.com/davitshahnazaryan3/SLFGenerator
    We acknowledge and appreciate Dr. Shahnazaryan’s original contribution.
    """
    
    def __init__(self,
                 component_data: component_data_model,
                 edp: str,
                 correlation_tree: correlation_tree_model = None,
                 typology: List[str] = None,
                 edp_range: Union[List[float], np.ndarray] = None,
                 edp_bin: float = None,
                 grouping_flag: bool = True,
                 conversion: float = 1.0,
                 realizations: int = 20,
                 replacement_cost: float = 1.0,
                 regression: str = "Weibull",
                 storey: Union[int, List[int]] = None,
                 directionality: int = None):
        """
        Initialize the SLF Generator.

        Parameters
        ----------
        component_data : component_data_model 
            Inventory of component data.  

        edp : str  
            Engineering demand parameter (EDP) options are : 'PSD' (Peak Storey Drift) or 'PFA' (Peak Floor Acceleration).  

        correlation_tree : correlation_tree_model, optional  
            Correlation tree for the component data. Default is None.  

        typology : List[str], optional  
            Type of components considered; options are: "ns" (Non-structural) or "s" (Structural). Default is None.  

        edp_range : Union[List[float], np.ndarray], optional  
            Range of EDP values. Default is None.  

        edp_bin : float, optional  
            Size of the EDP bin. Default is None.  

        grouping_flag : bool, optional  
            Whether to perform performance grouping of components. Default is True.  

        conversion : float, optional  
            Conversion factor for cost-related values.

        realizations : int, optional  
            Number of realizations for the Monte Carlo method. Default is 20.  

        replacement_cost : float, optional  
            Replacement cost of the building (used when normalizing SLFs). Default is 1.0.  

        regression : str, optional  
            Regression function to be used; supported options: 'Weibull' (default), 'Papadopoulos', 'Gdp' (Genpareto), and 'Lognormal'. 

        storey : Union[int, List[int]], optional  
            Storey levels to consider. Default is None.  

        directionality : int, optional  
            Directionality of the analysis. Default is None (None means non-directional).  
        """

        self.edp = edp.lower()
        self.typology = typology
        self.edp_bin = edp_bin
        self.edp_range = edp_range
        self.grouping_flag = grouping_flag
        self.conversion = conversion
        self.realizations = realizations
        self.replacement_cost = replacement_cost
        self.regression = regression.lower()
        self.storey = storey
        self.directionality = directionality
        self.correlation_tree = correlation_tree

        # Normalize component data
        self.component_data = component_data.applymap(lambda s: s.lower() if isinstance(s, str) else s)

        # Define EDP range and component data
        self._define_edp_range()
        self._get_component_data()

        # Handle correlation tree if provided
        if self.correlation_tree:
            self.correlation_tree.applymap(lambda s: s.lower() if isinstance(s, str) else s)
            self._get_correlation_tree()

        # Group components if required
        if self.grouping_flag:
            self._group_components()


    def _define_edp_range(self):
        """Define range of engineering demand parameters (EDP).
    
        Raises
        ------
        ValueError
            If incorrect EDP type is provided, must be 'psd' or 'pfa'.
        """
        edp_defaults = {
            "idr": (0.1 / 100, 0, 0.5),
            "psd": (0.1 / 100, 0, 0.5),
            "pfa": (0.05, 0, 5),
        }
    
        if self.edp not in edp_defaults:
            raise ValueError("Incorrect EDP is provided, must be either 'psd' or 'pfa'")
    
        default_bin, range_start, range_end = edp_defaults[self.edp]
        self.edp_bin = self.edp_bin if self.edp_bin is not None else default_bin
    
        if self.edp_range is None:
            self.edp_range = np.arange(range_start, range_end + self.edp_bin, self.edp_bin)
    
        self.edp_range[0] = 1e-20
        self.edp_range = np.asarray(self.edp_range)

    def _get_component_data(self):
        """Fetches and processes component data from the provided .csv file.
        Components with missing IDs will be assigned automatically.
        Newly created entries will not persist if the .csv file is modified.
        """
    
        # Validate component data schema
        self._validate_component_data_schema()
    
        # Handle 'best fit' columns (fill missing values with 'normal')
        best_fit_cols = [col for col in self.component_data if col.endswith("Best Fit")]
        self.component_data[best_fit_cols] = self.component_data[best_fit_cols].fillna("normal")
    
        # Assign default values for 'ITEM' and 'ID' columns
        self.component_data["Component ID"] = self.component_data["Component ID"].fillna(
            pd.Series(np.arange(1, len(self.component_data) + 1), dtype="int")
        )
        self.component_data["Description"] = self.component_data["Description"].fillna("B")
    
        # Fill missing values in all other columns except 'Group' and 'Component'
        exclude_cols = ["Performance Group", "Typology"]
        cols_to_fill = self.component_data.columns.difference(exclude_cols)
        self.component_data[cols_to_fill] = self.component_data[cols_to_fill].fillna(0)

    def _group_components(self):
        """Component performance grouping."""
        # Ensure placeholders for missing values
        self.component_data["Performance Group"].fillna(-1, inplace=True)
        self.component_data["Typology"].fillna("-1", inplace=True)
    
        # If no grouping is needed, assign all data under the first EDP value
        if not self.grouping_flag:
            key = self.component_data["EDP"].iloc[0]
            self.component_groups = {key: self.component_data}
            return
    
        # Perform grouping based on EDP and Component
        edp_groups = self.component_data.groupby(["EDP", "Typology"])
    
        # Define specific groups if applicable
        if "psd" in self.component_data["EDP"].values:
            psd_s = edp_groups.get_group(("psd", "s")) if ("psd", "s") in edp_groups.groups else None
            psd_ns = edp_groups.get_group(("psd", "ns")) if ("psd", "ns") in edp_groups.groups else None
        else:
            psd_s = psd_ns = None
    
        pfa_ns = edp_groups.get_group(("pfa", "ns")) if ("pfa", "ns") in edp_groups.groups else None
    
        # Assign the component groups dictionary
        self.component_groups = {
            "PSD, S": psd_s,
            "PSD, NS": psd_ns,
            "PFA, NS": pfa_ns
        }
    
        # Remove None values
        self.component_groups = {k: v for k, v in self.component_groups.items() if v is not None}
    
        # If Group exists, override default EDP-based grouping
        if self.component_data["Performance Group"].nunique() > 1:
            self.component_groups = {group: df for group, df in self.component_data.groupby("Performance Group")}


    def _get_correlation_tree(self) -> np.ndarray[int]:
        """
        Load a correlation tree from a .csv file.
        
        Notes on Correlation Tree Creation:  
        Ensure that the minimum damage state (MIN DS) assigned does not exceed the available damage states (DS) defined for each component. For example, if a dependent component has only one damage state (e.g., DS1), which occurs only when its causation element reaches DS3, then the dependency must be correctly defined. Additionally, if a component (e.g., Item3) is expected to sustain damage earlier and has more DS levels than another, this should be accurately reflected in the correlation structure.  
        Note: The software does not validate these dependencies automatically, so correctness relies on the user.
        
        Updates
        ----------
        matrix : np.ndarray [number of components × (number of damage states + 2)]  
            Correlation table defining relationships between component IDs.
        
        Examples
        ----------
            +---------------+-------------+-------------+-------------+
            | Component ID  | Dependent on | MIN DS | DS0  | MIN DS | DS1 |
            +===============+=============+=============+=============+
            | Item 1        | Independent | Independent | Independent |
            +---------------+-------------+-------------+-------------+
            | Item 2        | 1           | Undamaged   | Undamaged   |
            +---------------+-------------+-------------+-------------+
            | Item 3        | 1           | Undamaged   | Undamaged   |
            +---------------+-------------+-------------+-------------+
        
            Continued...
        
            +-------------+-------------+-------------+-------------+
            | MIN DS | DS2 | MIN DS | DS3 | MIN DS | DS4 | MIN DS | DS5 |
            +=============+=============+=============+=============+
            | Independent | Independent | Independent | Independent |
            +-------------+-------------+-------------+-------------+
            | Undamaged   | DS1         | DS1         | DS1         |
            +-------------+-------------+-------------+-------------+
            | DS1         | DS2         | DS3         | DS3         |
            +-------------+-------------+-------------+-------------+
        """

        # Extract damage states and correlation data
        damage_states = list(self.component_data["Damage States"])
        correlation_data = self.correlation_tree.loc[self.component_data.index].values
    
        # Validate correlation tree schema
        self._validate_correlation_tree_schema(damage_states)
    
        # Extract item IDs and remove the first column to get correlation values
        item_ids = correlation_data[:, 0]
        correlation_values = np.delete(correlation_data, 0, axis=1)
    
        # Initialize correlation matrix
        self.matrix = np.full(correlation_values.shape, np.nan, dtype=float)
    
        # Populate correlation matrix
        for i, row in enumerate(correlation_values):
            for j, value in enumerate(row):
                if j == 0:  # First column
                    if isinstance(value, str) and value.lower() == "independent":
                        self.matrix[i, j] = item_ids[i]
                    elif not item_ids[i] or math.isnan(item_ids[i]):
                        self.matrix[i, j] = np.nan
                    else:
                        self.matrix[i, j] = value
                else:  # Remaining columns
                    if math.isnan(self.matrix[i, j - 1]):
                        self.matrix[i, j] = np.nan
                    elif isinstance(value, str) and value.lower() in {"independent", "undamaged"}:
                        self.matrix[i, j] = 0
                    else:
                        self.matrix[i, j] = int(value[-1])

    def _validate_component_data_schema(self):
        columns = list(self.component_data.columns)
        component_data = self.component_data.to_dict(orient='records')

        # Validate base fields
        id_set = set()
        for row in component_data:
            model = component_data_model.parse_obj(row)
            if model.Component_ID is not None and model.Component_ID in id_set:
                raise ValueError(f'Duplicate ITEM: {model.Component_ID}')
            id_set.add(model.Component_ID)

        counts = {
            "Median": 0,
            "Total Dispersion": 0,
            "Cost": 0,
            "Cost Dispersion": 0,
            "Best Fit": 0,
        }

        for col in columns:
            for key in counts.keys():
                if col.endswith(key):
                    counts[key] += 1

        total_count = counts["Median"]
        for key in counts.keys():
            if total_count != counts[key]:
                raise ValueError(
                    "There must be equal amount of columns: 'Median', "
                    "'Total Dispersion, 'Cost', "
                    "'Cost Dispersion', 'Best Fit")

    def _validate_correlation_tree_schema(self, 
                                          damage_states):
        
        corr_dict = self.correlation_tree.to_dict(orient='records')

        # Validate base fields
        id_set = set()
        for row in corr_dict:
            model = correlation_tree_model.parse_obj(row)
            if model.ITEM in id_set:
                raise ValueError(f'Duplicate ITEM: {model.ITEM}')
            id_set.add(model.ITEM)

        # Check integrity of the provided input correlation table
        if len(self.correlation_tree.keys()) < max(damage_states) + 3:
            raise ValueError(
                "[EXCEPTION] Unexpected (fewer) number of features "
                "in the correlations DataFrame")

        # Verify integrity of the provided correlation tree
        idx = 0
        for item in self.component_data.index:
            for feature in self.correlation_tree.keys():
                ds = str(self.correlation_tree.loc[item][feature])
                if ds == f'DS{damage_states[idx] + 1}':
                    raise ValueError("[EXCEPTION] MIN DS assigned in "
                                     "the correlation tree must not exceed "
                                     "the possible DS defined for the element")
            idx += 1

        # Check that dimensions of the correlation tree
        # and the component data match
        if len(self.component_data) != len(self.correlation_tree):
            raise ValueError(
                "[EXCEPTION] Number of items in the correlation tree "
                "and component data should match")


    def fragility_function(self) -> tuple[dict, np.ndarray, np.ndarray]:
        """Derives fragility functions.
    
        Returns
        -------
        dict
            Fragility functions associated with each damage state and component.
        np.ndarray
            Mean values of cost functions (shape: number of components, number of damage states).
        np.ndarray
            Covariances of cost functions (shape: number of components, number of damage states).
        """
    
        # Count the number of Damage States (DS) using string matching
        n_ds = self.component_data.columns.str.endswith("Median").sum()
    
        # Extract numerical data excluding categorical columns
        data = self.component_data.select_dtypes(exclude=['object']).drop(
            labels=['Component ID', 'Performance Group', 'Quantity', 'Damage States'], axis=1
        ).values
    
        num_components = len(data)
    
        # Initialize fragility and consequence parameter arrays
        means_fr, covs_fr = data[:, :n_ds], data[:, n_ds:2*n_ds]
        means_cost, covs_cost = data[:, 2*n_ds:3*n_ds] * self.conversion, data[:, 3*n_ds:4*n_ds]
    
        # Construct fragility functions
        fragilities = {"EDP": self.edp_range, "IDs": {}}
    
        for item in range(num_components):
            fragilities["IDs"][item + 1] = {}
    
            for ds in range(n_ds):
                mean_val, cov_val = means_fr[item, ds], covs_fr[item, ds]
    
                if mean_val == 0:
                    fragilities["IDs"][item + 1][f"DS{ds + 1}"] = np.zeros(len(self.edp_range))
                else:
                    log_std = np.log(cov_val ** 2 + 1) ** 0.5
                    log_mean = np.exp(np.log(mean_val) - 0.5 * log_std**2)
                    fragility_curve = stats.norm.cdf(np.log(self.edp_range / log_mean) / log_std)
    
                    fragilities["IDs"][item + 1][f"DS{ds + 1}"] = np.nan_to_num(fragility_curve)
    
        return fragilities, means_cost, covs_cost


    def do_monte_carlo_simulations(self, 
                                        fragilities: fragility_model) -> ds_model:
        """Performs Monte Carlo simulations and simulates damage state (DS) for
        each engineering demand parameter (EDP) value
    
        Parameters
        ----------
        fragilities : fragility_model         Fragility functions of all components at all DSs
    
        Returns
        ----------
        ds_model                     Sampled damage states of each component for each simulation
        """
        # Number of damage states
        n_ds = len(fragilities['IDs'][1])
        ds_range = np.arange(0, n_ds + 1)
    
        # Prepare the random numbers once per simulation
        random_arrays = np.random.rand(self.realizations, len(self.edp_range))
    
        # Initialize damage_state dictionary
        damage_state = {}
    
        # Iterate over items in fragility model
        for item, frag in fragilities['IDs'].items():
            damage_state[item] = {}
    
            # For each simulation
            for n in range(self.realizations):
                random_array = random_arrays[n]  # Get pre-generated random array for this realization
                damage = np.zeros(len(self.edp_range), dtype=int)
    
                # For each DS, process in reverse order
                for ds in range(n_ds, 0, -1):
                    y1 = frag[f"DS{ds}"]
                    if ds == n_ds:
                        # Apply damage based on the fragility value for the last damage state
                        damage = np.where(random_array <= y1, ds_range[ds], damage)
                    else:
                        y2 = frag[f"DS{ds + 1}"]
                        damage = np.where((random_array >= y2) & (random_array < y1), ds_range[ds], damage)
    
                damage_state[item][n] = damage
    
        return damage_state

    def validate_ds_dependence(self, 
                             damage_state: ds_model) -> ds_model:
        """Enforces new DS for each dependent component

        Parameters
        ----------
        damage_state : ds_model          Sampled damage states of each component for each simulation

        Returns
        ----------
        ds_model                         Sampled DS of each component for each simulation after enforcing
                                                 DS for dependent components if a correlation matrix is provided
        """
        if self.correlation_tree is None:
            return damage_state

        for i in range(self.matrix.shape[0]):
            # Check if component is dependent or independent
            if i + 1 != self.matrix[i][0]:
                # -- Component is dependent
                # Causation component ID
                m = self.matrix[i][0]
                # Dependent component ID
                j = i + 1
                # Loop for each simulation
                for n in range(self.realizations):
                    causation_ds = damage_state[m][n]
                    correlated_ds = damage_state[j][n]

                    # Get dependent components DS conditioned
                    # on causation component
                    temp = np.zeros(causation_ds.shape)
                    # Loop over each DS
                    for ds in range(1, self.matrix.shape[1]):
                        temp[causation_ds == ds - 1] = self.matrix[j - 1][ds]

                    # Modify DS if correlated component is conditioned on
                    # causation component's DS, otherwise skip
                    damage_state[j][n] = np.maximum(correlated_ds, temp)

        return damage_state

    def calculate_costs(self,
                        damage_state: ds_model,
                        means_cost: np.ndarray,
                        covs_cost: np.ndarray,
                        ) -> tuple[cost_model, cost_model, simulation_model]:

        """
        Evaluates the damage cost on the individual i-th component at each
        EDP (Engineering Demand Parameter) level for each n-th simulation.

        This method calculates the total replacement costs and repair costs associated with each component
        based on the given damage states, mean cost values, and cost covariances. It provides a detailed cost 
        analysis for each component and simulation, which can be used for risk and loss assessments.

        Parameters
        ----------
        damage_state : ds_model
            Sampled damage states for each component, indicating the extent of damage at different levels.

        means_cost : np.ndarray (number of components, number of damage states)
            Mean values of the cost functions, representing the expected cost of repair or replacement 
            for each component at each damage state.

        covs_cost : np.ndarray (number of components, number of damage states)
            Covariances of the cost functions, indicating the uncertainty or variability of the cost values 
            for each component and damage state.

        Returns
        ----------
        cost_model
            Total replacement costs in absolute values, calculated for all components and damage states.

        cost_model
            Total replacement costs as a ratio of the replacement cost, providing a normalized cost estimate.

        simulation_model
            Repair costs associated with each component and simulation, showing the cost for each component 
            at each damage state during the simulation process.
        
        """        
        # Number of damage states
        num_ds = means_cost.shape[1]

        repair_cost = {}
        for item in damage_state.keys():
            idx = int(item) - 1
            repair_cost[item] = {}
            for n in range(self.realizations):
                for ds in range(num_ds + 1):
                    if ds == 0:
                        repair_cost[item][n] = np.where(
                            damage_state[item][n] == ds, ds, -1)

                    else:
                        # Best fit function
                        best_fit = \
                            self.component_data.iloc[
                                item - 1][f"DS{ds}, Best Fit"].lower()

                        # EDP ID where ds is observed
                        idx_list = np.where(damage_state[item][n] == ds)[0]
                        for idx_repair in idx_list:
                            if best_fit == 'lognormal':
                                a = np.random.normal(means_cost[idx][ds - 1],
                                                     covs_cost[idx][ds - 1]
                                                     * means_cost[idx][ds - 1])
                                while a < 0:
                                    std = covs_cost[idx][ds - 1] * \
                                        means_cost[idx][ds - 1]
                                    m = np.log(
                                        means_cost[idx][ds - 1] ** 2
                                        / np.sqrt(means_cost[idx][ds - 1] ** 2
                                                  + std ** 2))
                                    std_log = np.sqrt(np.log(
                                        (means_cost[idx][ds - 1] ** 2
                                         + std ** 2)
                                        / means_cost[idx][ds - 1] ** 2))
                                    a = np.random.lognormal(m, std_log)
                            else:
                                a = np.random.normal(means_cost[idx][ds - 1],
                                                     covs_cost[idx][ds - 1]
                                                     * means_cost[idx][ds - 1])
                                while a < 0:
                                    a = np.random.normal(
                                        means_cost[idx][ds - 1],
                                        covs_cost[idx][ds - 1]
                                        * means_cost[idx][ds - 1])

                            repair_cost[item][n][idx_repair] = a
            idx += 1

        # Evaluate the total damage cost multiplying the individual
        # cost by each element quantity
        quantities = self.component_data["Quantity"]
        total_repair_cost = {}
        idx = 0
        for item in damage_state.keys():
            total_repair_cost[item] = {}
            for n in range(self.realizations):
                total_repair_cost[item][n] = repair_cost[item][n] * \
                    quantities.iloc[item - 1]
            idx += 1

        # Evaluate total loss for the storey segment
        total_loss_storey = {}
        for n in range(self.realizations):
            total_loss_storey[n] = np.zeros(len(self.edp_range))
            for item in damage_state.keys():
                total_loss_storey[n] += total_repair_cost[item][n]

        # Calculate if replCost was set to 0, otherwise use the provided value
        if self.replacement_cost == 0.0 or self.replacement_cost is None:
            raise ValueError(
                "Replacement cost should be a non-negative non-zero value.")
        else:
            total_replacement_cost = self.replacement_cost

        total_loss_storey_ratio = {}
        for n in range(self.realizations):
            total_loss_storey_ratio[n] = total_loss_storey[n] / \
                total_replacement_cost

        return total_loss_storey, total_loss_storey_ratio, repair_cost

    def perform_regression(self, 
                           loss: cost_model, 
                           loss_ratio: cost_model, 
                           regression_type: str = None, 
                           percentiles: List[float] = None) -> tuple[loss_model,fitted_loss_model, fitting_parameters_model, float, float]:

        """
        Perform regression analysis on the loss and loss ratio data to estimate fitted loss functions.

        The function fits loss models to the loss and loss ratio data using various regression types, including 
        Weibull, Papadopoulos, Generalized Pareto Distribution (GPD), and Lognormal. If no specific regression 
        type is provided, it will try all available models and return the one with the best fit.

        Parameters
        ----------
        loss : cost_model
            DataFrame containing loss values for each component and damage state.
        
        loss_ratio : cost_model
            DataFrame containing loss ratio values for each component and damage state.

        regression_type : str, optional
            The regression model to be used. Supported options are "weibull", "papadopoulos", "gpd", and "lognormal".
            If None, all regression models are tried, and the best fitting model is selected. Default is None.

        percentiles : List[float], optional
            List of percentiles for which the loss and loss ratio values will be computed. Default is [0.16, 0.50, 0.84].

        Returns
        -------
        tuple
            A tuple containing:
            - `losses` (dict): Quantiles of the loss and loss ratio data for the specified percentiles.
            - `losses_fitted` (fitted_loss_model): The fitted loss function based on the selected regression model.
            - `fitting_parameters` (fitting_parameters_model): The parameters of the fitted loss function.
            - `error_max` (float): The maximum error of the regression as a percentage.
            - `error_cum` (float): The cumulative error of the regression as a percentage.

        Raises
        ------
        ValueError
            If the provided `regression_type` is not supported.
        """        
        # Set default percentiles if not provided
        percentiles = percentiles or [0.16, 0.50, 0.84]
    
        # Convert loss and loss_ratio to DataFrames
        loss = pd.DataFrame.from_dict(loss)
        loss_ratio = pd.DataFrame.from_dict(loss_ratio)
    
        # Compute quantiles and mean values for losses and loss ratios
        losses = {
            "loss": loss.quantile(percentiles, axis=1),
            "loss_ratio": loss_ratio.quantile(percentiles, axis=1)
        }
    
        mean_loss = loss.mean(axis=1)
        mean_loss_ratio = loss_ratio.mean(axis=1)
    
        # Set the 'mean' row for losses and loss_ratio
        losses["loss"].loc['mean'] = mean_loss
        losses["loss_ratio"].loc['mean'] = mean_loss_ratio
    
        # Determine edp_range based on self.edp
        edp_range = self.edp_range * 100 if self.edp in ["idr", "psd"] else self.edp_range
    
        # Define fitting functions and initial parameter guesses
        fitting_functions = {
            "weibull": {
                "func": lambda x, a, b, c: a * (1 - np.exp(-((x / b) ** c))),
                "p0": [1.0, 1.0, 1.0]  # Initial guess for a, b, c
            },
            "papadopoulos": {
                "func": lambda x, a, b, c, d, e: e * (x ** a) / (b ** a + x ** a) + (1 - e) * (x ** c) / (d ** c + x ** c),
                "p0": [1.0, 1.0, 1.0, 1.0, 0.5]  # Initial guess for a, b, c, d, e
            },
            "gpd": {
                "func": lambda x, c, loc, scale: genpareto.cdf(x, c, loc=loc, scale=scale),
                "p0": [0.1, 0.0, 1.0]  # Initial guess for c, loc, scale
            },
            "lognormal": {
                "func": lambda x, mean, sigma: lognorm.cdf(x, sigma, scale=np.exp(mean)),
                "p0": [1.0, 1.0]  # Initial guess for mean, sigma
            }
        }
    
        # If regression_type is None, try all regression models and select the best one
        if regression_type is None:
            best_losses_fitted = None
            best_fitting_parameters = None
            best_error_max = float('inf')
            best_error_cum = float('inf')
    
            for reg_type in fitting_functions.keys():
                try:
                    losses_fitted, fitting_parameters, error_max, error_cum = self._fit_regression(
                        losses, edp_range, fitting_functions[reg_type], percentiles
                    )
                    if error_max < best_error_max and error_cum < best_error_cum:
                        best_losses_fitted = losses_fitted
                        best_fitting_parameters = fitting_parameters
                        best_error_max = error_max
                        best_error_cum = error_cum
                except Exception as e:
                    print(f"Regression failed for {reg_type}: {e}")
    
            return losses, best_losses_fitted, best_fitting_parameters, best_error_max, best_error_cum
    
        # If a specific regression type is provided, use that
        if regression_type not in fitting_functions:
            raise ValueError(f"Regression type {regression_type} is not supported.")
        
        losses_fitted, fitting_parameters, error_max, error_cum = self._fit_regression(
            losses, edp_range, fitting_functions[regression_type], percentiles
        )
    
        return losses, losses_fitted, fitting_parameters, error_max, error_cum
    
    def _fit_regression(self, 
                        losses, 
                        edp_range, 
                        fitting_function, 
                        percentiles):
        
        losses_fitted = {}
        fitting_parameters = {}
    
        for q in percentiles + ['mean']:
            max_val = max(losses["loss_ratio"].loc[q])
            normalized_loss_ratio = losses["loss_ratio"].loc[q] / max_val
    
            try:
                popt, pcov = curve_fit(fitting_function["func"], edp_range, normalized_loss_ratio, p0=fitting_function["p0"], maxfev=10**6)
            except RuntimeError as e:
                print(f"Regression failed for {fitting_function} at quantile {q}: {e}")
                continue
    
            fitted_loss = fitting_function["func"](edp_range, *popt) * max_val
            fitted_loss[fitted_loss <= 0] = 0.0
    
            losses_fitted[q] = fitted_loss
            fitting_parameters[q] = {"popt": popt, "pcov": pcov}
    
        error_max, error_cum = self.estimate_accuracy(losses["loss_ratio"].loc['mean'], losses_fitted['mean'])
    
        return losses_fitted, fitting_parameters, error_max, error_cum


    def estimate_accuracy(self, y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
        """
        Estimate the prediction accuracy by calculating the maximum and cumulative errors as a percentage 
        relative to the maximum observed value.

        Parameters
        ----------
        y : np.ndarray
            Observations or true values.
        
        yhat : np.ndarray
            Predicted values.

        Returns
        -------
        tuple[float, float]
            A tuple containing:
            - Maximum error in percentage (%), calculated as the largest absolute error relative to the maximum observed value.
            - Cumulative error in percentage (%), calculated as the sum of the absolute errors weighted by the bin size relative to the maximum observed value.
        """
        # Ensure inputs are numpy arrays
        y, yhat = np.asarray(y), np.asarray(yhat)
        
        # Calculate absolute error and the maximum of y
        abs_error = np.abs(y - yhat)
        max_y = np.max(y)
        
        # Compute the errors
        error_max = np.max(abs_error) / max_y * 100
        error_cum = self.edp_bin * np.sum(abs_error) / max_y * 100
    
        return error_max, error_cum


    def transform_output(self, 
                         losses_fitted: fitted_loss_model, 
                         typology: str = None) -> slf_model:
        """
        Transforms the fitted storey-loss-function (SLF) output to primary attributes supported by
        the Loss Assessment module.

        This method organizes the fitted loss functions into a structured format, including key parameters
        such as directionality, component type, storey, and engineering demand parameter (EDP), which 
        are required for further analysis and loss assessments.

        Parameters
        ----------
        losses_fitted : fitted_loss_model
            Fitted loss functions containing the mean values of the storey loss functions.

        typology : str, optional
            Type of component considered in the analysis. Default is None.

        Returns
        -------
        slf_model
            A dictionary containing the SLF output with primary attributes, such as directionality, 
            component type, storey, EDP, and fitted loss function data.
        """
        out = {
            'Directionality': self.directionality,
            'Component-type': typology,
            'Storey': self.storey,
            'edp': self.edp,
            'edp_range': list(self.edp_range),
            'slf': list(losses_fitted['mean']),
        }

        return out

    def generate(self):
        """Generate Storey Loss Functions (SLFs) for each performance group.

        This method computes the Storey Loss Functions (SLFs) for different performance groups based on fragility functions, 
        Monte Carlo simulations for damage state sampling, regression analysis, and empirical statistics. 
        The results are stored for each group and returned along with a cache of relevant data for future use.

        The process involves:
        - Obtaining component fragility and consequence functions.
        - Performing Monte Carlo simulations for sampling damage states.
        - Validating damage state dependencies.
        - Calculating costs for each component group.
        - Performing regression analysis to fit loss functions.
        - Transforming the results into the required SLF format.
        - Computing empirical statistics (median, 16th percentile, 84th percentile) for each group.
        
        Returns
        -------
        tuple
            A tuple containing:
            - `out` (Dict[slf_model]): A dictionary where the key is the group identifier (as a string) and the value is the 
              Storey Loss Function (SLF) for that group, including the fitted loss functions and their associated errors.
            - `cache` (Dict): A dictionary storing intermediate data such as component data, fragility functions, total losses, 
              repair costs, damage states, and the results of the regression analysis for each group.
        """        
        
        # Initialise dictionaries
        out, cache = {}, {}
    
        # Obtain component fragility and consequence functions
        fragilities, means_cost, covs_cost = self.fragility_function()
    
        # Perform Monte Carlo simulations for damage state sampling
        damage_state = self.do_monte_carlo_simulations(fragilities)
    
        # Populate the damage state matrix for correlated components
        damage_state = self.validate_ds_dependence(damage_state)
    
        # Iterate over component groups
        for group, component_data_group in self.component_groups.items():
            if component_data_group.empty:
                continue
    
            # Determine component type
            typology = (
                self.typology[group].lower()
                if isinstance(self.typology, dict)
                else (
                    self.typology[0].lower()
                    if isinstance(self.typology, list) and len(self.typology) > 0
                    else None
                )
            )
    
            # Prepare group data
            item_ids = list(component_data_group['Component ID'])
            ds_group = {key: damage_state[key] for key in item_ids}
            fragilities_group = {
                'IDs': {key: fragilities['IDs'][key] for key in item_ids},
                'EDP': fragilities['EDP']
            }
    
            # Calculate the costs
            total, ratio, repair_cost = self.calculate_costs(ds_group, means_cost, covs_cost)
    
            # Perform regression (if regression_type is None, it will try all models)
            losses, losses_fitted, fitting_parameters, error_max, error_cum = self.perform_regression(total, ratio, self.regression)
    
            # Transform output and store results
            group_str = str(group)
            out[group_str] = self.transform_output(losses_fitted, typology)
            out[group_str]['error_max'], out[group_str]['error_cum'] = error_max, error_cum

            # Cache relevant data for future use
            cache[group_str] = {
                'component': component_data_group,
                'fragilities': fragilities_group,
                'total_loss_storey': total,
                'total_loss_storey_ratio': ratio,
                'repair_cost': repair_cost,
                'damage_states': damage_state,
                'losses': losses,
                'slfs': losses_fitted,
                'fit_pars': fitting_parameters,
                'accuracy': [error_max, error_cum],
                'regression': self.regression,
                'edp': self.edp
            }

            # Compute empirical statistics along axis 0 (column-wise for each EDP range)
            median = np.median(np.array([cache[group_str]['total_loss_storey'][i] for i in range(len(cache[group_str]['total_loss_storey']))]), axis=0)
            percentile_16th = np.percentile(np.array([cache[group_str]['total_loss_storey'][i] for i in range(len(cache[group_str]['total_loss_storey']))]), 16, axis=0)
            percentile_84th = np.percentile(np.array([cache[group_str]['total_loss_storey'][i] for i in range(len(cache[group_str]['total_loss_storey']))]), 84, axis=0)

            # Recache relevant data for future use
            cache[group_str] = {
                'component': component_data_group,
                'fragilities': fragilities_group,
                'total_loss_storey': total,
                'total_loss_storey_ratio': ratio,
                'repair_cost': repair_cost,
                'damage_states': damage_state,
                'losses': losses,
                'slfs': losses_fitted,
                'fit_pars': fitting_parameters,
                'accuracy': [error_max, error_cum],
                'regression': self.regression,
                'edp': self.edp,
                'empirical_median': median,
                'empirical_16th': percentile_16th,
                'empirical_84th': percentile_84th
            }

        self.cache = cache
                
        return out, cache
    
