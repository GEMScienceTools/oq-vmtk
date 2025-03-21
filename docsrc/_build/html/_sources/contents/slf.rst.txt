Storey-Loss Function Generation Module
######################################

The ``slf_generator`` module provides a class ``slf_generator`` to generate Storey Loss Functions
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

   .. method:: __init__(component_data, edp, correlation_tree=None, typology=None, edp_range=None, edp_bin=None, grouping_flag=True, conversion=1.0, realizations=20, replacement_cost=1.0, regression="Weibull", storey=None, directionality=None)

      Initializes the SLF Generator with the provided parameters.

      :param component_data: Inventory of component data.
      :type component_data: component_data_model
      :param edp: Engineering Demand Parameter (EDP) options are: "psd" (Peak Storey Drift) or "pfa" (Peak Floor Acceleration).
      :type edp: str
      :param correlation_tree: Correlation tree for the component data. Default is `None`.
      :type correlation_tree: correlation_tree_model, optional
      :param typology: Type of components considered; options are: "ns" (Non-structural) or "s" (Structural). Default is `None`.
      :type typology: List[str], optional
      :param edp_range: Range of EDP values. Default is `None`.
      :type edp_range: Union[List[float], np.ndarray], optional
      :param edp_bin: Size of the EDP bin. Default is `None`.
      :type edp_bin: float, optional
      :param grouping_flag: Whether to perform performance grouping of components. Default is `True`.
      :type grouping_flag: bool, optional
      :param conversion: Conversion factor for cost-related values. Default is `1.0`.
      :type conversion: float, optional
      :param realizations: Number of realizations for the Monte Carlo method. Default is `20`.
      :type realizations: int, optional
      :param replacement_cost: Replacement cost of the building (used when normalizing SLFs). Default is `1.0`.
      :type replacement_cost: float, optional
      :param regression: Regression function to be used. Supported options: "Weibull" (default), "Papadopoulos", "Gdp" (Generalized Pareto Distribution), and "Lognormal".
      :type regression: str, optional
      :param storey: Storey levels to consider. Default is `None`.
      :type storey: Union[int, List[int]], optional
      :param directionality: Directionality of the analysis. Default is `None` (non-directional).
      :type directionality: int, optional

   .. method:: _define_edp_range()
      Defines the range of Engineering Demand Parameters (EDP) based on the provided EDP type.

   .. method:: _get_component_data()
      Fetches and processes component data from the provided input.

   .. method:: _group_components()
      Groups components based on performance and typology if `grouping_flag` is `True`.

   .. method:: _get_correlation_tree()
      Loads and processes the correlation tree if provided.

   .. method:: fragility_function()
      Derives fragility functions for each component based on the provided data.

      :return: Fragility functions associated with each damage state and component.
      :rtype: dict
      :return: Mean values of cost functions.
      :rtype: np.ndarray
      :return: Covariances of cost functions.
      :rtype: np.ndarray

   .. method:: do_monte_carlo_simulations(fragilities)
      Performs Monte Carlo simulations to sample damage states for each component.

      :param fragilities: Fragility functions of all components at all damage states.
      :type fragilities: fragility_model
      :return: Sampled damage states of each component for each simulation.
      :rtype: ds_model

   .. method:: validate_ds_dependence(damage_state)
      Validates damage state dependencies based on the correlation tree.

      :param damage_state: Sampled damage states of each component for each simulation.
      :type damage_state: ds_model
      :return: Sampled damage states after enforcing dependencies.
      :rtype: ds_model

   .. method:: calculate_costs(damage_state, means_cost, covs_cost)
      Calculates repair and replacement costs for each component based on the sampled damage states.

      :param damage_state: Sampled damage states for each component.
      :type damage_state: ds_model
      :param means_cost: Mean values of the cost functions.
      :type means_cost: np.ndarray
      :param covs_cost: Covariances of the cost functions.
      :type covs_cost: np.ndarray
      :return: Total replacement costs in absolute values.
      :rtype: cost_model
      :return: Total replacement costs as a ratio of the replacement cost.
      :rtype: cost_model
      :return: Repair costs associated with each component and simulation.
      :rtype: simulation_model

   .. method:: perform_regression(loss, loss_ratio, regression_type=None, percentiles=[0.16, 0.50, 0.84])
      Performs regression analysis on the loss and loss ratio data to estimate fitted loss functions.

      :param loss: DataFrame containing loss values for each component and damage state.
      :type loss: cost_model
      :param loss_ratio: DataFrame containing loss ratio values for each component and damage state.
      :type loss_ratio: cost_model
      :param regression_type: The regression model to be used. Supported options: "Weibull", "Papadopoulos", "Gdp", and "Lognormal". Default is `None`.
      :type regression_type: str, optional
      :param percentiles: List of percentiles for which the loss and loss ratio values will be computed. Default is `[0.16, 0.50, 0.84]`.
      :type percentiles: List[float], optional
      :return: Quantiles of the loss and loss ratio data.
      :rtype: loss_model
      :return: The fitted loss function based on the selected regression model.
      :rtype: fitted_loss_model
      :return: The parameters of the fitted loss function.
      :rtype: fitting_parameters_model
      :return: The maximum error of the regression as a percentage.
      :rtype: float
      :return: The cumulative error of the regression as a percentage.
      :rtype: float

   .. method:: estimate_accuracy(y, yhat)
      Estimates the prediction accuracy by calculating the maximum and cumulative errors as a percentage relative to the maximum observed value.

      :param y: Observations or true values.
      :type y: np.ndarray
      :param yhat: Predicted values.
      :type yhat: np.ndarray
      :return: Maximum error in percentage.
      :rtype: float
      :return: Cumulative error in percentage.
      :rtype: float

   .. method:: transform_output(losses_fitted, typology=None)
      Transforms the fitted Storey Loss Function (SLF) output into a structured format.

      :param losses_fitted: Fitted loss functions containing the mean values of the storey loss functions.
      :type losses_fitted: fitted_loss_model
      :param typology: Type of component considered in the analysis. Default is `None`.
      :type typology: str, optional
      :return: A dictionary containing the SLF output with primary attributes.
      :rtype: slf_model

   .. method:: generate()
      Generates Storey Loss Functions (SLFs) for each performance group.

      :return: A dictionary where the key is the group identifier and the value is the SLF for that group.
      :rtype: Dict[slf_model]
      :return: A dictionary storing intermediate data such as component data, fragility functions, total losses, repair costs, damage states, and regression results.
      :rtype: Dict

Example Usage
-------------

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

1. Ramirez, C. and Miranda, E., (2009) "Building-specific loss estimation methods
   and tools for simplified performance-based earthquake engineering", John A. Blume
   Earthquake Engineering Center, Department of Civil and Environmental Engineering,
   Stanford University.

2. Shahnazaryan, D., O'Reilly, G.J., Monteiro R. "Story loss functions for seismic
   design and assessment: Development of tools and application," Earthquake Spectra 2021;
   37(4): 2813–2839. DOI: 10.1177/87552930211023523.

3. Shahnazaryan, D., O'Reilly, G.J., Monteiro R. "Development of a Python-Based
   Storey Loss Function Generator," COMPDYN 2021 - 8th International Conference on
   Computational Methods in Structural Dynamics and Earthquake Engineering, 2021.
   DOI: 10.7712/120121.8659.18567.
