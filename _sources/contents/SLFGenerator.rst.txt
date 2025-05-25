Storey-Loss Function Generation Module
######################################

The `slf_generator` module provides a class ``slf_generator`` to generate Storey Loss Functions
(SLFs) based on fragility, consequence, and quantity data. SLFs establish a direct relationship
between the expected loss at a specific storey and the engineering demand parameter.
This class employs a probabilistic approach, utilizing Monte Carlo simulations to model damage,
assess the associated loss, and determine its distribution within a storey, considering a
user-defined inventory of damageable components.

Classes
-------

.. class:: slf_generator()

  A class for generating Storey Loss Functions (SLFs) using fragility, consequence, and quantity data. It applies a probabilistic approach to quantify the loss and its distribution across various storeys of a building under seismic loading.

  **Attributes**:
  - **edp**: `str`
    The Engineering Demand Parameter (EDP) for the analysis (e.g., "psd" for Peak Storey Drift or "pfa" for Peak Floor Acceleration).

  - **typology**: `List[str]`
    The type of components considered (e.g., "structural" or "non-structural").

  - **edp_bin**: `float`
    The size of the EDP bin used for discretizing the EDP range.

  - **edp_range**: `Union[List[float], np.ndarray]`
    The range of EDP values over which the SLFs are calculated.

  - **grouping_flag**: `bool`
    Whether to perform performance grouping of components. Default is `True`.

  - **conversion**: `float`
    Conversion factor for cost-related values. Default is `1.0`.

  - **realizations**: `int`
    Number of realizations for the Monte Carlo method. Default is `20`.

  - **replacement_cost**: `float`
    Replacement cost of the building (used when normalizing SLFs). Default is `1.0`.

  - **regression**: `str`
    Regression function to be used for fitting the loss functions. Supported options are "Weibull" (default), "Papadopoulos", "Gdp" (Generalized Pareto Distribution), and "Lognormal".

  - **storey**: `Union[int, List[int]]`
    Storey levels to consider in the analysis. Default is `None`.

  - **directionality**: `int`
    Directionality of the analysis. Default is `None` (non-directional).

  - **correlation_tree**: `correlation_tree_model`
    Correlation tree for the component data. Default is `None`.

  **Methods**:
  - **\_\_init\_\_**:
    Initializes the SLF Generator with the provided parameters.

    **Parameters**:
    - **component_data**: `component_data_model`
      Inventory of component data.
    - **edp**: `str`
      Engineering Demand Parameter (EDP) options are: "psd" (Peak Storey Drift) or "pfa" (Peak Floor Acceleration).
    - **correlation_tree**: `correlation_tree_model`, optional
      Correlation tree for the component data. Default is `None`.
    - **typology**: `List[str]`, optional
      Type of components considered; options are: "ns" (Non-structural) or "s" (Structural). Default is `None`.
    - **edp_range**: `Union[List[float], np.ndarray]`, optional
      Range of EDP values. Default is `None`.
    - **edp_bin**: `float`, optional
      Size of the EDP bin. Default is `None`.
    - **grouping_flag**: `bool`, optional
      Whether to perform performance grouping of components. Default is `True`.
    - **conversion**: `float`, optional
      Conversion factor for cost-related values. Default is `1.0`.
    - **realizations**: `int`, optional
      Number of realizations for the Monte Carlo method. Default is `20`.
    - **replacement_cost**: `float`, optional
      Replacement cost of the building (used when normalizing SLFs). Default is `1.0`.
    - **regression**: `str`, optional
      Regression function to be used. Supported options: "Weibull" (default), "Papadopoulos", "Gdp" (Generalized Pareto Distribution), and "Lognormal".
    - **storey**: `Union[int, List[int]]`, optional
      Storey levels to consider. Default is `None`.
    - **directionality**: `int`, optional
      Directionality of the analysis. Default is `None` (non-directional).

  - **\_define_edp_range**:
    Defines the range of Engineering Demand Parameters (EDP) based on the provided EDP type.

  - **\_get_component_data**:
    Fetches and processes component data from the provided input.

  - **\_group_components**:
    Groups components based on performance and typology if `grouping_flag` is `True`.

  - **\_get_correlation_tree**:
    Loads and processes the correlation tree if provided.

  - **fragility_function**:
    Derives fragility functions for each component based on the provided data.

    **Returns**:
    - `dict`: Fragility functions associated with each damage state and component.
    - `np.ndarray`: Mean values of cost functions.
    - `np.ndarray`: Covariances of cost functions.

  - **do_monte_carlo_simulations**:
    Performs Monte Carlo simulations to sample damage states for each component.

    **Parameters**:
    - **fragilities**: `fragility_model`
      Fragility functions of all components at all damage states.

    **Returns**:
    - `ds_model`: Sampled damage states of each component for each simulation.

  - **validate_ds_dependence**:
    Validates damage state dependencies based on the correlation tree.

    **Parameters**:
    - **damage_state**: `ds_model`
      Sampled damage states of each component for each simulation.

    **Returns**:
    - `ds_model`: Sampled damage states after enforcing dependencies.

  - **calculate_costs**:
    Calculates repair and replacement costs for each component based on the sampled damage states.

    **Parameters**:
    - **damage_state**: `ds_model`
      Sampled damage states for each component.
    - **means_cost**: `np.ndarray`
      Mean values of the cost functions.
    - **covs_cost**: `np.ndarray`
      Covariances of the cost functions.

    **Returns**:
    - `cost_model`: Total replacement costs in absolute values.
    - `cost_model`: Total replacement costs as a ratio of the replacement cost.
    - `simulation_model`: Repair costs associated with each component and simulation.

  - **perform_regression**:
    Performs regression analysis on the loss and loss ratio data to estimate fitted loss functions.

    **Parameters**:
    - **loss**: `cost_model`
      DataFrame containing loss values for each component and damage state.
    - **loss_ratio**: `cost_model`
      DataFrame containing loss ratio values for each component and damage state.
    - **regression_type**: `str`, optional
      The regression model to be used. Supported options: "Weibull", "Papadopoulos", "Gdp", and "Lognormal". Default is `None`.
    - **percentiles**: `List[float]`, optional
      List of percentiles for which the loss and loss ratio values will be computed. Default is `[0.16, 0.50, 0.84]`.

    **Returns**:
    - `loss_model`: Quantiles of the loss and loss ratio data.
    - `fitted_loss_model`: The fitted loss function based on the selected regression model.
    - `fitting_parameters_model`: The parameters of the fitted loss function.
    - `float`: The maximum error of the regression as a percentage.
    - `float`: The cumulative error of the regression as a percentage.

  - **estimate_accuracy**:
    Estimates the prediction accuracy by calculating the maximum and cumulative errors as a percentage relative to the maximum observed value.

    **Parameters**:
    - **y**: `np.ndarray`
      Observations or true values.
    - **yhat**: `np.ndarray`
      Predicted values.

    **Returns**:
    - `float`: Maximum error in percentage.
    - `float`: Cumulative error in percentage.

  - **transform_output**:
    Transforms the fitted Storey Loss Function (SLF) output into a structured format.

    **Parameters**:
    - **losses_fitted**: `fitted_loss_model`
      Fitted loss functions containing the mean values of the storey loss functions.
    - **typology**: `str`, optional
      Type of component considered in the analysis. Default is `None`.

    **Returns**:
    - `slf_model`: A dictionary containing the SLF output with primary attributes.

  - **generate**:
    Generates Storey Loss Functions (SLFs) for each performance group.

    **Returns**:
    - `Dict[slf_model]`: A dictionary where the key is the group identifier and the value is the SLF for that group.
    - `Dict`: A dictionary storing intermediate data such as component data, fragility functions, total losses, repair costs, damage states, and regression results.

Example Usage
------------

.. code-block:: python

    from slf_generator import slf_generator

    # Example component data
    component_data = pd.read_csv('inventory.csv')

    # Initialize SLF Generator
    model = slf_generator(component_data=component_data,
                          edp="psd",
                          typology=["structural"],
                          edp_range=[0.0, 0.5],
                          edp_bin=0.1,
                          realizations=20,
                          replacement_cost=1000000.0,
                          regression="Weibull")

    # Generate SLFs
    out, cache = model.generate()

    # Access the results
    print(out)  # Fitted SLFs
    print(cache)  # Intermediate data and empirical statistics

References
----------
1) Ramirez, C. and Miranda, E., (2009) "Building-specific loss estimation methods
and tools for simplified performance-based earthquake engineering", John A. Blume
Earthquake Engineering Center, Department of Civil and Environmental Engineering,
Stanford University.

2) Shahnazaryan, D., O'Reilly, G.J., Monteiro R. "Story loss functions for seismic
design and assessment: Development of tools and application," Earthquake Spectra 2021;
37(4): 2813–2839. DOI: 10.1177/87552930211023523.

3) Shahnazaryan, D., O'Reilly, G.J., Monteiro R. "Development of a Python-Based
torey Loss Function Generator," COMPDYN 2021 - 8th International Conference on
Computational Methods in Structural Dynamics and Earthquake Engineering, 2021.
DOI: 10.7712/120121.8659.18567.
