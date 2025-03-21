Plotter Module
##############

The ``plotter`` class is a utility class for creating and customizing various types of
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

.. class:: plotter

   .. method:: __init__()
      Initializes the ``plotter`` class with default styles for fonts, line widths, marker sizes, colors, resolution, and font name.

   .. method:: _set_plot_style(ax, title=None, xlabel=None, ylabel=None, grid=True)
      Sets consistent plot style for all plots.

      :param ax: The axes object to apply the style to.
      :type ax: matplotlib.axes.Axes
      :param title: The title of the plot. Default is `None`.
      :type title: str, optional
      :param xlabel: The label for the x-axis. Default is `None`.
      :type xlabel: str, optional
      :param ylabel: The label for the y-axis. Default is `None`.
      :type ylabel: str, optional
      :param grid: Whether to display the grid. Default is `True`.
      :type grid: bool, optional

   .. method:: _save_plot(output_directory, plot_label)
      Saves the plot to the specified directory.

      :param output_directory: Directory where the plot will be saved. If `None`, the plot is not saved.
      :type output_directory: str, optional
      :param plot_label: The label for the saved plot file (without file extension).
      :type plot_label: str

   .. method:: duplicate_for_drift(peak_drift_list, control_nodes)
      Creates data for box plots of peak storey drifts.

      :param peak_drift_list: A list of arrays containing peak drift values for each floor.
      :type peak_drift_list: list of np.ndarray
      :param control_nodes: A list of control nodes (floors) in the structure.
      :type control_nodes: list
      :return: Processed x and y data for plotting.
      :rtype: list

   .. method:: plot_cloud_analysis(cloud_dict, output_directory=None, plot_label='cloud_analysis_plot', xlabel='Peak Ground Acceleration, PGA [g]', ylabel=r'Maximum Peak Storey Drift, $\theta_{max}$ [%]')
      Plots cloud analysis results, including scatter points, regression line, and censoring limits.

      :param cloud_dict: A dictionary containing the data for the cloud analysis.
      :type cloud_dict: dict
      :param output_directory: Directory where the plot will be saved. Default is `None`.
      :type output_directory: str, optional
      :param plot_label: The label for the saved plot file. Default is `'cloud_analysis_plot'`.
      :type plot_label: str, optional
      :param xlabel: The label for the x-axis. Default is `'Peak Ground Acceleration, PGA [g]'`.
      :type xlabel: str, optional
      :param ylabel: The label for the y-axis. Default is `'Maximum Peak Storey Drift, $\theta_{max}$ [%]'`.
      :type ylabel: str, optional

   .. method:: plot_fragility_analysis(cloud_dict, output_directory=None, plot_label='fragility_plot', xlabel='Peak Ground Acceleration, PGA [g]')
      Plots fragility analysis results, showing the probability of exceedance for various damage states.

      :param cloud_dict: A dictionary containing the data for the fragility analysis.
      :type cloud_dict: dict
      :param output_directory: Directory where the plot will be saved. Default is `None`.
      :type output_directory: str, optional
      :param plot_label: The label for the saved plot file. Default is `'fragility_plot'`.
      :type plot_label: str, optional
      :param xlabel: The label for the x-axis. Default is `'Peak Ground Acceleration, PGA [g]'`.
      :type xlabel: str, optional

   .. method:: plot_demand_profiles(peak_drift_list, peak_accel_list, control_nodes, output_directory=None, plot_label='demand_profiles')
      Plots demand profiles for peak drifts and accelerations.

      :param peak_drift_list: A list of arrays containing peak drift values for each floor.
      :type peak_drift_list: list of np.ndarray
      :param peak_accel_list: A list of arrays containing peak acceleration values for each floor.
      :type peak_accel_list: list of np.ndarray
      :param control_nodes: A list of control nodes (floors) in the structure.
      :type control_nodes: list
      :param output_directory: Directory where the plot will be saved. Default is `None`.
      :type output_directory: str, optional
      :param plot_label: The label for the saved plot file. Default is `'demand_profiles'`.
      :type plot_label: str, optional

   .. method:: plot_ansys_results(cloud_dict, peak_drift_list, peak_accel_list, control_nodes, output_directory=None, plot_label='ansys_results', cloud_xlabel='PGA', cloud_ylabel='MPSD')
      Plots a 2x2 grid of analysis results, including cloud analysis, fragility analysis, and demand profiles.

      :param cloud_dict: A dictionary containing the data for the cloud and fragility analyses.
      :type cloud_dict: dict
      :param peak_drift_list: A list of arrays containing peak drift values for each floor.
      :type peak_drift_list: list of np.ndarray
      :param peak_accel_list: A list of arrays containing peak acceleration values for each floor.
      :type peak_accel_list: list of np.ndarray
      :param control_nodes: A list of control nodes (floors) in the structure.
      :type control_nodes: list
      :param output_directory: Directory where the plot will be saved. Default is `None`.
      :type output_directory: str, optional
      :param plot_label: The label for the saved plot file. Default is `'ansys_results'`.
      :type plot_label: str, optional
      :param cloud_xlabel: The label for the x-axis of the cloud analysis plot. Default is `'PGA'`.
      :type cloud_xlabel: str, optional
      :param cloud_ylabel: The label for the y-axis of the cloud analysis plot. Default is `'MPSD'`.
      :type cloud_ylabel: str, optional

   .. method:: plot_vulnerability_analysis(intensities, loss, cov, xlabel, ylabel, output_directory=None, plot_label='vulnerability_plot')
      Plots vulnerability analysis results, including Beta distributions and loss curves.

      :param intensities: A list of intensity measures (e.g., Peak Ground Acceleration, PGA).
      :type intensities: list of float
      :param loss: A list of mean loss ratios corresponding to each intensity measure.
      :type loss: list of float
      :param cov: A list of coefficients of variation (CoV) corresponding to each intensity measure.
      :type cov: list of float
      :param xlabel: The label for the x-axis.
      :type xlabel: str
      :param ylabel: The label for the y-axis.
      :type ylabel: str
      :param output_directory: Directory where the plot will be saved. Default is `None`.
      :type output_directory: str, optional
      :param plot_label: The label for the saved plot file. Default is `'vulnerability_plot'`.
      :type plot_label: str, optional

   .. method:: plot_slf_model(out, cache, xlabel, output_directory=None, plot_label='slf')
      Plots Storey Loss Function (SLF) model results.

      :param out: A dictionary containing the results of the model.
      :type out: dict
      :param cache: A dictionary containing cached data, including total storey losses and empirical statistics.
      :type cache: dict
      :param xlabel: The label for the x-axis.
      :type xlabel: str
      :param output_directory: Directory where the plot will be saved. Default is `None`.
      :type output_directory: str, optional
      :param plot_label: The label for the saved plot file. Default is `'slf'`.
      :type plot_label: str, optional

   .. method:: animate_model_run(control_nodes, acc, dts, nrha_disps, nrha_accels, drift_thresholds, output_directory=None, plot_label='animation')
      Animates the seismic demands for a single nonlinear time-history analysis (NRHA) run.

      :param control_nodes: A list of nodes (floors) in the model.
      :type control_nodes: list
      :param acc: A 1D array of acceleration values corresponding to the time-history of seismic excitation.
      :type acc: numpy.ndarray
      :param dts: A 1D array of time steps (in seconds) for the NRHA analysis.
      :type dts: numpy.ndarray
      :param nrha_disps: A 2D array of node displacements (in meters) for each time step and control node.
      :type nrha_disps: numpy.ndarray
      :param nrha_accels: A 2D array of node accelerations (in g) for each time step and control node.
      :type nrha_accels: numpy.ndarray
      :param drift_thresholds: A list of drift thresholds that define the damage states for the nodes in the model.
      :type drift_thresholds: list
      :param output_directory: Directory where the animation will be saved. Default is `None`.
      :type output_directory: str, optional
      :param plot_label: The label for the saved animation file. Default is `'animation'`.
      :type plot_label: str, optional
