Plotter Module
##############

The `plotter` class is a utility class for creating and customizing various types of
plots for structural analysis results. It provides methods to visualize data from
structural analyses, including cloud analysis, fragility analysis, demand profiles,
vulnerability analysis, and animations of seismic responses. The class also includes
utility methods for setting consistent plot styles and saving plots.

**Attributes**:
- **font_sizes**: `dict`
  Dictionary containing font sizes for titles, labels, ticks, and legends.

- **line_widths**: `dict`
  Dictionary containing line widths for thick, medium, and thin lines.

- **marker_sizes**: `dict`
  Dictionary containing marker sizes for large, medium, and small markers.

- **colors**: `dict`
  Dictionary containing color schemes for fragility, damage states, and GEM colors.

- **resolution**: `int`
  Resolution for saving plots (default: 500 DPI).

- **font_name**: `str`
  Font name for plot text (default: 'Arial').

**Methods**:
- **\_\_init\_\_**:
  Initializes the `plotter` class with default styles for fonts, line widths, marker sizes, colors, resolution, and font name.

- **_set_plot_style(ax, title=None, xlabel=None, ylabel=None, grid=True)**:
  Sets consistent plot style for all plots.

  **Parameters**:
  - **ax**: `matplotlib.axes.Axes`
    The axes object to apply the style to.
  - **title**: `str`, optional
    The title of the plot. Default is `None`.
  - **xlabel**: `str`, optional
    The label for the x-axis. Default is `None`.
  - **ylabel**: `str`, optional
    The label for the y-axis. Default is `None`.
  - **grid**: `bool`, optional
    Whether to display the grid. Default is `True`.

- **_save_plot(output_directory, plot_label)**:
  Saves the plot to the specified directory.

  **Parameters**:
  - **output_directory**: `str`, optional
    Directory where the plot will be saved. If `None`, the plot is not saved.
  - **plot_label**: `str`
    The label for the saved plot file (without file extension).

- **duplicate_for_drift(peak_drift_list, control_nodes)**:
  Creates data for box plots of peak storey drifts.

  **Parameters**:
  - **peak_drift_list**: `list of np.ndarray`
    A list of arrays containing peak drift values for each floor.
  - **control_nodes**: `list`
    A list of control nodes (floors) in the structure.

  **Returns**:
  - `list`: Processed x and y data for plotting.

- **plot_cloud_analysis(cloud_dict, output_directory=None, plot_label='cloud_analysis_plot', xlabel='Peak Ground Acceleration, PGA [g]', ylabel=r'Maximum Peak Storey Drift, $\theta_{max}$ [%]')**:
  Plots cloud analysis results, including scatter points, regression line, and censoring limits.

  **Parameters**:
  - **cloud_dict**: `dict`
    A dictionary containing the data for the cloud analysis.
  - **output_directory**: `str`, optional
    Directory where the plot will be saved. Default is `None`.
  - **plot_label**: `str`, optional
    The label for the saved plot file. Default is `'cloud_analysis_plot'`.
  - **xlabel**: `str`, optional
    The label for the x-axis. Default is `'Peak Ground Acceleration, PGA [g]'`.
  - **ylabel**: `str`, optional
    The label for the y-axis. Default is `'Maximum Peak Storey Drift, $\theta_{max}$ [%]'`.

- **plot_fragility_analysis(cloud_dict, output_directory=None, plot_label='fragility_plot', xlabel='Peak Ground Acceleration, PGA [g]')**:
  Plots fragility analysis results, showing the probability of exceedance for various damage states.

  **Parameters**:
  - **cloud_dict**: `dict`
    A dictionary containing the data for the fragility analysis.
  - **output_directory**: `str`, optional
    Directory where the plot will be saved. Default is `None`.
  - **plot_label**: `str`, optional
    The label for the saved plot file. Default is `'fragility_plot'`.
  - **xlabel**: `str`, optional
    The label for the x-axis. Default is `'Peak Ground Acceleration, PGA [g]'`.

- **plot_demand_profiles(peak_drift_list, peak_accel_list, control_nodes, output_directory=None, plot_label='demand_profiles')**:
  Plots demand profiles for peak drifts and accelerations.

  **Parameters**:
  - **peak_drift_list**: `list of np.ndarray`
    A list of arrays containing peak drift values for each floor.
  - **peak_accel_list**: `list of np.ndarray`
    A list of arrays containing peak acceleration values for each floor.
  - **control_nodes**: `list`
    A list of control nodes (floors) in the structure.
  - **output_directory**: `str`, optional
    Directory where the plot will be saved. Default is `None`.
  - **plot_label**: `str`, optional
    The label for the saved plot file. Default is `'demand_profiles'`.

- **plot_ansys_results(cloud_dict, peak_drift_list, peak_accel_list, control_nodes, output_directory=None, plot_label='ansys_results', cloud_xlabel='PGA', cloud_ylabel='MPSD')**:
  Plots a 2x2 grid of analysis results, including cloud analysis, fragility analysis, and demand profiles.

  **Parameters**:
  - **cloud_dict**: `dict`
    A dictionary containing the data for the cloud and fragility analyses.
  - **peak_drift_list**: `list of np.ndarray`
    A list of arrays containing peak drift values for each floor.
  - **peak_accel_list**: `list of np.ndarray`
    A list of arrays containing peak acceleration values for each floor.
  - **control_nodes**: `list`
    A list of control nodes (floors) in the structure.
  - **output_directory**: `str`, optional
    Directory where the plot will be saved. Default is `None`.
  - **plot_label**: `str`, optional
    The label for the saved plot file. Default is `'ansys_results'`.
  - **cloud_xlabel**: `str`, optional
    The label for the x-axis of the cloud analysis plot. Default is `'PGA'`.
  - **cloud_ylabel**: `str`, optional
    The label for the y-axis of the cloud analysis plot. Default is `'MPSD'`.

- **plot_vulnerability_analysis(intensities, loss, cov, xlabel, ylabel, output_directory=None, plot_label='vulnerability_plot')**:
  Plots vulnerability analysis results, including Beta distributions and loss curves.

  **Parameters**:
  - **intensities**: `list of float`
    A list of intensity measures (e.g., Peak Ground Acceleration, PGA).
  - **loss**: `list of float`
    A list of mean loss ratios corresponding to each intensity measure.
  - **cov**: `list of float`
    A list of coefficients of variation (CoV) corresponding to each intensity measure.
  - **xlabel**: `str`
    The label for the x-axis.
  - **ylabel**: `str`
    The label for the y-axis.
  - **output_directory**: `str`, optional
    Directory where the plot will be saved. Default is `None`.
  - **plot_label**: `str`, optional
    The label for the saved plot file. Default is `'vulnerability_plot'`.

- **plot_slf_model(out, cache, xlabel, output_directory=None, plot_label='slf')**:
  Plots Storey Loss Function (SLF) model results.

  **Parameters**:
  - **out**: `dict`
    A dictionary containing the results of the model.
  - **cache**: `dict`
    A dictionary containing cached data, including total storey losses and empirical statistics.
  - **xlabel**: `str`
    The label for the x-axis.
  - **output_directory**: `str`, optional
    Directory where the plot will be saved. Default is `None`.
  - **plot_label**: `str`, optional
    The label for the saved plot file. Default is `'slf'`.

- **animate_model_run(control_nodes, acc, dts, nrha_disps, nrha_accels, drift_thresholds, output_directory=None, plot_label='animation')**:
  Animates the seismic demands for a single nonlinear time-history analysis (NRHA) run.

  **Parameters**:
  - **control_nodes**: `list`
    A list of nodes (floors) in the model.
  - **acc**: `numpy.ndarray`
    A 1D array of acceleration values corresponding to the time-history of seismic excitation.
  - **dts**: `numpy.ndarray`
    A 1D array of time steps (in seconds) for the NRHA analysis.
  - **nrha_disps**: `numpy.ndarray`
    A 2D array of node displacements (in meters) for each time step and control node.
  - **nrha_accels**: `numpy.ndarray`
    A 2D array of node accelerations (in g) for each time step and control node.
  - **drift_thresholds**: `list`
    A list of drift thresholds that define the damage states for the nodes in the model.
  - **output_directory**: `str`, optional
    Directory where the animation will be saved. Default is `None`.
  - **plot_label**: `str`, optional
    The label for the saved animation file. Default is `'animation'`.
