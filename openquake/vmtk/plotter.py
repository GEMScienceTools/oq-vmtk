import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
    
class plotter:
    """
    A class for creating and customizing various types of plots for structural analysis results.

    This class provides methods to visualize data from structural analyses, including cloud analysis,
    fragility analysis, demand profiles, vulnerability analysis, and animations of seismic responses.
    It also includes utility methods for setting consistent plot styles and saving plots.

    Attributes
    ----------
    font_sizes : dict
        Dictionary containing font sizes for titles, labels, ticks, and legends.
    line_widths : dict
        Dictionary containing line widths for thick, medium, and thin lines.
    marker_sizes : dict
        Dictionary containing marker sizes for large, medium, and small markers.
    colors : dict
        Dictionary containing color schemes for fragility, damage states, and GEM colors.
    resolution : int
        Resolution for saving plots (default: 500 DPI).
    font_name : str
        Font name for plot text (default: 'Arial').

    Methods
    -------
    _set_plot_style(ax, title=None, xlabel=None, ylabel=None, grid=True)
        Sets consistent plot style for all plots.
    _save_plot(output_directory, plot_label)
        Saves the plot to the specified directory.
    duplicate_for_drift(peak_drift_list, control_nodes)
        Creates data for box plots of peak storey drifts.
    plot_cloud_analysis(cloud_dict, output_directory=None, plot_label='cloud_analysis_plot', xlabel='Peak Ground Acceleration, PGA [g]', ylabel=r'Maximum Peak Storey Drift, $\theta_{max}$ [%]')
        Plots cloud analysis results.
    plot_fragility_analysis(cloud_dict, output_directory=None, plot_label='fragility_plot', xlabel='Peak Ground Acceleration, PGA [g]')
        Plots fragility analysis results.
    plot_demand_profiles(peak_drift_list, peak_accel_list, control_nodes, output_directory=None, plot_label='demand_profiles')
        Plots demand profiles for peak drifts and accelerations.
    plot_ansys_results(cloud_dict, peak_drift_list, peak_accel_list, control_nodes, output_directory=None, plot_label='ansys_results', cloud_xlabel='PGA', cloud_ylabel='MPSD')
        Plots a 2x2 grid of analysis results, including cloud, fragility, and demand profiles.
    plot_vulnerability_analysis(intensities, loss, cov, xlabel, ylabel, output_directory=None, plot_label='vulnerability_plot')
        Plots vulnerability analysis results, including Beta distributions and loss curves.
    plot_slf_model(out, cache, xlabel, output_directory=None, plot_label='slf')
        Plots Storey Loss Function (SLF) model results.
    animate_model_run(control_nodes, acc, dts, nrha_disps, nrha_accels, drift_thresholds, output_directory=None, plot_label='animation')
        Animates the seismic demands for a single nonlinear time-history analysis (NRHA) run.

    Notes
    -----
    - The class uses Matplotlib and Seaborn for plotting.
    - The `_set_plot_style` method ensures consistent styling across all plots.
    - The `_save_plot` method handles saving plots with high resolution.
    - The `plot_cloud_analysis`, `plot_fragility_analysis`, and `plot_demand_profiles` methods are used for visualizing structural analysis results.
    - The `plot_vulnerability_analysis` method visualizes loss distributions and loss curves.
    - The `plot_slf_model` method visualizes Storey Loss Function (SLF) results.
    - The `animate_model_run` method creates animations of seismic responses.

    """    

    def __init__(self):
        # Define default styles
        self.font_sizes = {
            'title': 16,
            'labels': 14,
            'ticks': 12,
            'legend': 14
        }
        self.line_widths = {
            'thick': 3,
            'medium': 2,
            'thin': 1
        }
        self.marker_sizes = {
            'large': 100,
            'medium': 60,
            'small': 10
        }
        self.colors = {
            'fragility': ['green', 'yellow', 'orange', 'red'],
            'damage_states': ['blue', 'green', 'yellow', 'orange', 'red'],
            'gem': ["#0A4F4E", "#0A4F5E", "#54D7EB", "#54D6EB", "#399283", "#399264", "#399296"]
        }
        self.resolution = 500
        self.font_name = 'Arial'

    def _set_plot_style(self, ax, title=None, xlabel=None, ylabel=None, grid=True):
        """Set consistent plot style for all plots."""
        if title:
            ax.set_title(title, fontsize=self.font_sizes['title'], fontname=self.font_name)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.font_sizes['labels'], fontname=self.font_name)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.font_sizes['labels'], fontname=self.font_name)
        ax.tick_params(axis='both', labelsize=self.font_sizes['ticks'])
        if grid:
            ax.grid(visible=True, which='major')
            ax.grid(visible=True, which='minor')

    def _save_plot(self, output_directory, plot_label):
        """Save the plot if output_directory is provided."""
        if output_directory:
            plt.savefig(f'{output_directory}/{plot_label}.png', dpi=self.resolution, format='png')
        plt.show()

    def duplicate_for_drift(self, 
                            peak_drift_list, 
                            control_nodes):
        """Creates data to process box plots for peak storey drifts."""  
        x = []; y = []
        for i in range(len(control_nodes)-1):
            y.extend((float(control_nodes[i]),float(control_nodes[i+1])))
            x.extend((peak_drift_list[i],peak_drift_list[i]))
        y.append(float(control_nodes[i+1]))
        x.append(0.0)
        
        return x, y

    def plot_cloud_analysis(self, 
                            cloud_dict, 
                            output_directory=None, 
                            plot_label='cloud_analysis_plot',
                            xlabel='Peak Ground Acceleration, PGA [g]', 
                            ylabel=r'Maximum Peak Storey Drift, $\theta_{max}$ [%]'):
        
        """
        Generate a cloud analysis plot with scatter points and regression line, 
        visualizing the relationship between Peak Ground Acceleration (PGA) 
        and Maximum Peak Storey Drift.
    
        This method plots cloud data, damage thresholds, a fitted regression line, 
        and upper and lower censoring limits. The data is presented in logarithmic 
        scale for both axes.
    
        Parameters:
        ----------
        cloud_dict : dict
            A dictionary containing the data for the cloud analysis. The dictionary 
            should have the following keys (direct output from do_cloud_analysis method)
    
        output_directory : str, optional
            Directory where the plot will be saved. If None, the plot is saved 
            in the current working directory.
    
        plot_label : str, optional
            The label for the saved plot file (without file extension). Default is 
            'cloud_analysis_plot'.
    
        xlabel : str, optional
            The label for the x-axis. Default is 'Peak Ground Acceleration, PGA [g]'.
    
        ylabel : str, optional
            The label for the y-axis. Default is 'Maximum Peak Storey Drift, $\theta_{max}$ [%]'.
    
        Returns:
        --------
        None
            This function saves the plot to a file in the specified output directory.

        """       
        
        fig, ax = plt.subplots(figsize=(6, 6))
        self._set_plot_style(ax, xlabel=xlabel, ylabel=ylabel)

        ax.scatter(cloud_dict['cloud inputs']['imls'], cloud_dict['cloud inputs']['edps'], color=self.colors['gem'][2], s=self.marker_sizes['medium'], alpha=0.5, label='Cloud Data', zorder=0)
        for i in range(len(cloud_dict['cloud inputs']['damage_thresholds'])):
            ax.scatter(cloud_dict['fragility']['medians'][i], cloud_dict['cloud inputs']['damage_thresholds'][i], color=self.colors['fragility'][i], s=self.marker_sizes['large'], alpha=1.0, zorder=2)

        ax.plot(cloud_dict['regression']['fitted_x'], cloud_dict['regression']['fitted_y'], linestyle='solid', color=self.colors['gem'][1], lw=self.line_widths['thick'], label='Cloud Regression', zorder=1)
        ax.plot([min(cloud_dict['cloud inputs']['imls']), max(cloud_dict['cloud inputs']['imls'])], [cloud_dict['cloud inputs']['upper_limit'], cloud_dict['cloud inputs']['upper_limit']], '--', color=self.colors['gem'][-1], label='Upper Censoring Limit')
        ax.plot([min(cloud_dict['cloud inputs']['imls']), max(cloud_dict['cloud inputs']['imls'])], [cloud_dict['cloud inputs']['lower_limit'], cloud_dict['cloud inputs']['lower_limit']], '-.', color=self.colors['gem'][-1], label='Lower Censoring Limit')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([min(cloud_dict['cloud inputs']['imls']), max(cloud_dict['cloud inputs']['imls'])])
        ax.set_ylim([min(cloud_dict['cloud inputs']['edps']), max(cloud_dict['cloud inputs']['edps'])])
        ax.legend(fontsize=self.font_sizes['legend'])

        self._save_plot(output_directory, plot_label)

    def plot_fragility_analysis(self, 
                                cloud_dict, 
                                output_directory=None, 
                                plot_label='fragility_plot',
                                xlabel='Peak Ground Acceleration, PGA [g]'):
        
        """
        Generate a fragility analysis plot showing the probability of exceedance (PoE)
        for various damage states as a function of Peak Ground Acceleration (PGA).
    
        This method plots fragility curves for multiple damage states based on the 
        fragility data in the input dictionary. Each curve represents the probability 
        of exceedance for a specific damage state, and the plot is presented in a 
        linear scale for both axes.
    
        Parameters:
        ----------
        cloud_dict : dict
            A dictionary containing the data for the fragility analysis. The dictionary 
            should have the following keys:
                - 'fragility': A dictionary containing:
                    - 'intensities': List or array of intensity values (e.g., PGA levels).
                    - 'poes': 2D array of probabilities of exceedance for each damage state.
                - 'medians': List of medians for each damage state.
        
        output_directory : str, optional
            Directory where the plot will be saved. If None, the plot is saved 
            in the current working directory.
    
        plot_label : str, optional
            The label for the saved plot file (without file extension). Default is 
            'fragility_plot'.
    
        xlabel : str, optional
            The label for the x-axis. Default is 'Peak Ground Acceleration, PGA [g]'.
    
        Returns:
        --------
        None
            This function saves the plot to a file in the specified output directory.
        
        """
        
        fig, ax = plt.subplots(figsize=(6, 6))
        self._set_plot_style(ax, xlabel=xlabel, ylabel='Probability of Exceedance')

        for i in range(len(cloud_dict['fragility']['medians'])):
            ax.plot(cloud_dict['fragility']['intensities'], cloud_dict['fragility']['poes'][:, i], linestyle='solid', color=self.colors['fragility'][i], lw=self.line_widths['thick'], label=f'DS{i+1}')

        ax.set_xlim([0, 5])
        ax.set_ylim([0, 1])
        ax.legend(fontsize=self.font_sizes['legend'])

        self._save_plot(output_directory, plot_label)

    def plot_demand_profiles(self, 
                             peak_drift_list, 
                             peak_accel_list, 
                             control_nodes, 
                             output_directory=None, 
                             plot_label='demand_profiles'):
        
        """
        Generate demand profile plots for peak storey drifts and peak floor accelerations.
    
        This method creates two side-by-side plots:
        - A plot of peak storey drift (%), displaying how the drift varies with floor number.
        - A plot of peak floor acceleration (g), displaying how the acceleration varies with floor number.
        
        The data is presented as lines representing each control node's response at different floors.
    
        Parameters:
        ----------
        peak_drift_list : list of np.ndarray
            A list of arrays where each array contains peak drift values for each floor, with the first column being the drift values and the second column being the floor numbers.
    
        peak_accel_list : list of np.ndarray
            A list of arrays where each array contains peak acceleration values for each floor, with the first column being the acceleration values and the second column being the floor numbers.
    
        control_nodes : list
            A list of floor numbers or nodes that represent the control points in the structure.
    
        output_directory : str, optional
            Directory where the plot will be saved. If None, the plot is saved in the current working directory.
    
        plot_label : str, optional
            The label for the saved plot file (without file extension). Default is 'demand_profiles'.
    
        Returns:
        --------
        None
            This function saves the plot to a file in the specified output directory.
            
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self._set_plot_style(ax1, xlabel=r'Peak Storey Drift, $\theta_{max}$ [%]', ylabel='Floor No.')
        self._set_plot_style(ax2, xlabel=r'Peak Floor Acceleration, $a_{max}$ [g]', ylabel='Floor No.')

        nst = len(control_nodes) - 1
        for i in range(len(peak_drift_list)):
            x, y = self.duplicate_for_drift(peak_drift_list[i][:, 0], control_nodes)
            ax1.plot([float(i) * 100 for i in x], y, linewidth=self.line_widths['medium'], linestyle='solid', color=self.colors['gem'][1], alpha=0.7)
            ax2.plot([float(x) / 9.81 for x in peak_accel_list[i][:, 0]], control_nodes, linewidth=self.line_widths['medium'], linestyle='solid', color=self.colors['gem'][0], alpha=0.7)

        ax1.set_yticks(np.linspace(0, nst, nst + 1), labels=np.linspace(0, nst, nst + 1), minor=False)
        ax2.set_yticks(np.linspace(0, nst, nst + 1), labels=np.linspace(0, nst, nst + 1), minor=False)
        ax1.set_xticks(np.linspace(0, 5, 11), labels=np.linspace(0, 5, 11), minor=False)
        ax2.set_xticks(np.linspace(0, 5, 11), labels=np.linspace(0, 5, 11), minor=False)
        ax1.set_xlim([0, 5.0])
        ax2.set_xlim([0, 5.0])

        self._save_plot(output_directory, plot_label)


    def plot_ansys_results(self, 
                           cloud_dict, 
                           peak_drift_list, 
                           peak_accel_list, 
                           control_nodes, 
                           output_directory=None, 
                           plot_label='ansys_results',
                           cloud_xlabel='PGA', 
                           cloud_ylabel='MPSD'):
        """
        Generate a 2x2 grid of plots to visualize analysis results, including cloud analysis,
        fragility analysis, and demand profiles for both peak drifts and peak accelerations.
    
        This function generates four plots in a 2x2 grid layout:
        1. **Cloud Analysis**: Scatter plot of cloud data, fitted regression line, and censoring limits.
        2. **Fragility Analysis**: Plot of probability of exceedance (PoE) for different damage states.
        3. **Demand Profiles for Drifts**: Plot of peak storey drift (%) versus floor number.
        4. **Demand Profiles for Accelerations**: Plot of peak floor acceleration (g) versus floor number.
    
        Each plot is customized with appropriate labels, legends, and color schemes for clarity.
    
        Parameters:
        ----------
        cloud_dict : dict
            A dictionary containing the data for the cloud and fragility analyses. The dictionary should contain:
                - 'imls': Intensity Measure Levels for cloud analysis.
                - 'edps': Engineering Demand Parameters for cloud analysis.
                - 'cloud inputs': Dictionary with damage thresholds, upper and lower limits.
                - 'fragility': Dictionary with fragility intensities and probabilities of exceedance.
                - 'regression': Fitted x and y values for the cloud regression line.
                - 'medians': List of median values for each damage state.
        
        peak_drift_list : list of np.ndarray
            A list of arrays where each array contains peak drift values for each floor. The first column should be the drift values and the second column the floor numbers.
    
        peak_accel_list : list of np.ndarray
            A list of arrays where each array contains peak acceleration values for each floor. The first column should be the acceleration values and the second column the floor numbers.
    
        control_nodes : list
            A list of control node (floor) numbers for the structure.
    
        output_directory : str, optional
            Directory where the plot will be saved. If None, the plot is saved in the current working directory.
    
        plot_label : str, optional
            The label for the saved plot file (without file extension). Default is 'ansys_results'.
    
        cloud_xlabel : str, optional
            The label for the x-axis of the cloud analysis plot. Default is 'PGA'.
    
        cloud_ylabel : str, optional
            The label for the y-axis of the cloud analysis plot. Default is 'MPSD'.
    
        Returns:
        --------
        None
            This function saves the 2x2 grid of plots to a file in the specified output directory.
            
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        plt.rcParams['axes.axisbelow'] = True

        # Cloud Analysis
        self._set_plot_style(ax1, xlabel=cloud_xlabel, ylabel=cloud_ylabel)
        ax1.scatter(cloud_dict['cloud inputs']['imls'], cloud_dict['cloud inputs']['edps'], color=self.colors['gem'][2], s=self.marker_sizes['medium'], alpha=0.5, label='Cloud Data', zorder=0)
        for i in range(len(cloud_dict['cloud inputs']['damage_thresholds'])):
            ax1.scatter(cloud_dict['fragility']['medians'][i], cloud_dict['cloud inputs']['damage_thresholds'][i], color=self.colors['fragility'][i], s=self.marker_sizes['large'], alpha=1.0, zorder=2)
        ax1.plot(cloud_dict['regression']['fitted_x'], cloud_dict['regression']['fitted_y'], linestyle='solid', color=self.colors['gem'][1], lw=self.line_widths['thick'], label='Cloud Regression', zorder=1)
        ax1.plot([min(cloud_dict['cloud inputs']['imls']), max(cloud_dict['cloud inputs']['imls'])], [cloud_dict['cloud inputs']['upper_limit'], cloud_dict['cloud inputs']['upper_limit']], '--', color=self.colors['gem'][-1], label='Upper Censoring Limit')
        ax1.plot([min(cloud_dict['cloud inputs']['imls']), max(cloud_dict['cloud inputs']['imls'])], [cloud_dict['cloud inputs']['lower_limit'], cloud_dict['cloud inputs']['lower_limit']], '-.', color=self.colors['gem'][-1], label='Lower Censoring Limit')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend(fontsize=self.font_sizes['legend'])

        # Fragility Analysis
        self._set_plot_style(ax2, xlabel=cloud_xlabel, ylabel='Probability of Exceedance')
        for i in range(len(cloud_dict['fragility']['medians'])):
            ax2.plot(cloud_dict['fragility']['intensities'], cloud_dict['fragility']['poes'][:, i], linestyle='solid', color=self.colors['fragility'][i], lw=self.line_widths['thick'], label=f'DS{i+1}')
        ax2.set_xlim([0, 5])
        ax2.set_ylim([0, 1])
        ax2.legend(fontsize=self.font_sizes['legend'])

        # Demand Profiles: Drifts
        self._set_plot_style(ax3, xlabel=r'Peak Storey Drift, $\theta_{max}$ [%]', ylabel='Floor No.')
        nst = len(control_nodes) - 1
        for i in range(len(peak_drift_list)):
            x, y = self.duplicate_for_drift(peak_drift_list[i][:, 0], control_nodes)
            ax3.plot([float(i) * 100 for i in x], y, linewidth=self.line_widths['medium'], linestyle='solid', color=self.colors['gem'][1], alpha=0.7)
        ax3.set_yticks(np.linspace(0, nst, nst + 1), labels=np.linspace(0, nst, nst + 1), minor=False)
        ax3.set_xticks(np.linspace(0, 5, 11), labels=np.linspace(0, 5, 11), minor=False)
        ax3.set_xlim([0, 5.0])

        # Demand Profiles: Accelerations
        self._set_plot_style(ax4, xlabel=r'Peak Floor Acceleration, $a_{max}$ [g]', ylabel='Floor No.')
        for i in range(len(peak_accel_list)):
            ax4.plot([float(x) for x in peak_accel_list[i][:, 0]], control_nodes, linewidth=self.line_widths['medium'], linestyle='solid', color=self.colors['gem'][0], alpha=0.3)
        ax4.set_yticks(np.linspace(0, nst, nst + 1), labels=np.linspace(0, nst, nst + 1), minor=False)
        ax4.set_xticks(np.linspace(0, 5, 11), labels=np.linspace(0, 5, 11), minor=False)
        ax4.set_xlim([0, 5.0])

        plt.tight_layout()
        self._save_plot(output_directory, plot_label)

    def plot_vulnerability_analysis(self, 
                                    intensities, 
                                    loss, 
                                    cov, 
                                    xlabel, 
                                    ylabel, 
                                    output_directory=None, 
                                    plot_label='vulnerability_plot'):
        """
        Generate a plot to visualize the vulnerability analysis results, including 
        Beta distributions and a loss curve.
    
        This function simulates Beta distributions for each intensity measure using the 
        mean loss and coefficient of variation (CoV) provided, then visualizes these 
        distributions as violin plots with an overlaid strip plot. It also plots the 
        loss curve on a secondary y-axis, showing the relationship between intensity 
        and the loss ratio.
    
        The plot includes:
        1. A violin plot representing the Beta distributions for each intensity measure.
        2. A strip plot for better visualization of individual data points within the distributions.
        3. A loss curve plotted on a secondary y-axis to show the loss ratio as a function of intensity.
    
        Parameters:
        ----------
        intensities : list of float
            A list of intensity measures (e.g., Peak Ground Acceleration, PGA) for which 
            the vulnerability analysis is performed.
    
        loss : list of float
            A list of mean loss ratios corresponding to each intensity measure.
    
        cov : list of float
            A list of coefficients of variation (CoV) corresponding to each intensity measure 
            that will be used to simulate the Beta distributions.
    
        xlabel : str
            The label for the x-axis, typically representing the intensity measure (e.g., 'PGA').
    
        ylabel : str
            The label for the y-axis representing the loss curve, typically describing the loss ratio.
    
        output_directory : str, optional
            Directory where the plot will be saved. If None, the plot is saved in the current working directory.
    
        plot_label : str, optional
            The label for the saved plot file (without file extension). Default is 'vulnerability_plot'.
    
        Returns:
        --------
        None
            This function saves the plot to a file in the specified output directory.
        
        """
        # Simulating Beta distributions for each intensity measure
        simulated_data = []
        intensity_labels = []        
        for j, mean_loss in enumerate(loss):
            
            # Calculate variance using CoV
            variance = (cov[j] * mean_loss) ** 2  
            alpha = mean_loss * (mean_loss * (1 - mean_loss) / variance - 1)
            beta_param = (1 - mean_loss) * (mean_loss * (1 - mean_loss) / variance - 1)
            
            # Generate samples from the Beta distribution
            data = np.random.beta(alpha, beta_param, 10000)
            simulated_data.append(data)
            intensity_labels.extend([intensities[j]] * len(data))  # Repeat intensity measures for each sample
        
        # Convert to DataFrame for seaborn visualization
        df_sns = pd.DataFrame({
            'Intensity Measure': intensity_labels,
            'Simulated Data': np.concatenate(simulated_data)
        })
                    
        # Create a figure and a set of axes for the violin plot
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # --- Violin plot for Beta distributions ---
        violin=sns.violinplot(
                x='Intensity Measure', y='Simulated Data', data=df_sns,
                scale='width', bw=0.2, inner=None, ax=ax1, zorder=1
                )
        
        # Overlay a strip plot for better visualization of individual samples
        sns.stripplot(
            x='Intensity Measure', y='Simulated Data', data=df_sns,
            color='k', size=1, alpha=0.5, ax=ax1, zorder=3
        )
        
        # Customize the first y-axis (for the violin plot)
        ax1.set_ylabel("Simulated Loss Ratio", color='blue')
        ax1.set_xlabel(f"{xlabel}")
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.set_ylim(-0.1, 1.2)  # Adjust y-axis range for the violin plot
        
        # Add the legend for the violin plots (Beta distribution)
        # Create a dummy plot handle for the legend, since the violins are not directly plotted as lines
        beta_patch = mpatches.Patch(color=violin.collections[0].get_facecolor()[0], label="Beta Distribution")
        ax1.legend(handles=[beta_patch], loc='upper left', bbox_to_anchor=(0, 1), ncol=1)

        # --- Add a second set of x and y axes for the Loss Curve ---
        ax2 = ax1.twinx()  # Create a shared y-axis for the loss curve
        
        # Plot the loss curve on ax2 (now in blue)
        ax2.plot(
            range(len(intensities)), loss, marker='o', linestyle='-', color='blue',
            label="Loss Curve", zorder=2
        )
        
        # Customize the second y-axis (for the loss curve)
        ax2.set_ylabel(f"{ylabel}", color='blue', rotation = 270, labelpad=20)
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim(-0.1, 1.2)  # Adjust y-axis range for the loss curve if needed
        
        # Customize both x-axes to match
        ax1.set_xticks(range(len(intensities)))
        ax1.set_xticklabels([f"{x:.3f}" for x in intensities], rotation=45, ha='right')
                    
        # Add a legend for the loss curve
        ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.95), ncol=1)
        
        # Tight layout and show the combined plot
        plt.tight_layout()

        self._save_plot(output_directory, plot_label)


    def plot_slf_model(self, 
                       out, 
                       cache, 
                       xlabel, 
                       output_directory=None, 
                       plot_label='slf'):
        
        """
        Generate a plot to visualize the Storey Loss Function (SLF) model output.
    
        This function visualizes the storey loss for different realizations of a model 
        by plotting the following:
        1. Scatter plot of total storey loss for each realization.
        2. Shaded region representing the 16th to 84th percentiles of the empirical data.
        3. Plot of the median of the empirical data for simulations.
        4. Fitted Storey Loss Function (SLF) curve.
    
        The plot includes:
        - A scatter plot of the total loss per storey for each realization.
        - A shaded area representing the empirical 16th to 84th percentiles.
        - The median storey loss curve based on simulations.
        - The fitted SLF curve.
    
        Parameters:
        ----------
        out : dict
            A dictionary containing the results of the model. It should include keys for:
                - 'edp_range': A range of Engineering Demand Parameters (EDP) used in the analysis.
                - 'slf': The fitted Storey Loss Function curve.
        
        cache : dict
            A dictionary containing cached data, including:
                - 'total_loss_storey': A list of total storey losses for each realization.
                - 'empirical_16th', 'empirical_84th': Empirical data representing the 16th and 84th percentiles.
                - 'empirical_median': Empirical median values of the storey loss for the simulations.
    
        xlabel : str
            The label for the x-axis, typically representing the Engineering Demand Parameter (EDP) range.
    
        output_directory : str, optional
            Directory where the plot will be saved. If None, the plot is saved in the current working directory.
    
        plot_label : str, optional
            The label for the saved plot file (without file extension). Default is 'slf'.
    
        Returns:
        --------
        None
            This function saves the generated plot for each key in the `cache` dictionary to the specified directory.
        """        
        keys_list = list(cache.keys())
        for i, current_key in enumerate(keys_list):
            rlz = len(cache[current_key]['total_loss_storey'])
            total_loss_storey_array = np.array([cache[current_key]['total_loss_storey'][i] for i in range(rlz)])

            fig, ax = plt.subplots(figsize=(8, 6))
            self._set_plot_style(ax, xlabel=xlabel, ylabel='Storey Loss')

            for i in range(rlz):
                ax.scatter(out[current_key]['edp_range'], total_loss_storey_array[i, :], color=self.colors['gem'][3], s=self.marker_sizes['small'], alpha=0.5)

            ax.fill_between(out[current_key]['edp_range'], cache[current_key]['empirical_16th'], cache[current_key]['empirical_84th'], color='gray', alpha=0.3, label=r'16$^{\text{th}}$-84$^{\text{th}}$ Percentile')
            ax.plot(out[current_key]['edp_range'], cache[current_key]['empirical_median'], lw=self.line_widths['medium'], color='blue', label='Simulations - Median')
            ax.plot(out[current_key]['edp_range'], out[current_key]['slf'], color='black', lw=self.line_widths['medium'], label='SLF - Fitted')

            ax.legend(fontsize=self.font_sizes['legend'])
            self._save_plot(output_directory, f"{plot_label}_{current_key}")
            
    def animate_model_run(self, 
                          control_nodes, 
                          acc, 
                          dts, 
                          nrha_disps, 
                          nrha_accels, 
                          drift_thresholds, 
                          output_directory=None, 
                          plot_label='animation'):
        """
        Animate the seismic demands for a single nonlinear time-history analysis (NRHA) run.
    
        This function creates an animation that visualizes the time history of seismic responses, including:
        1. Floor displacement over time for each control node.
        2. Floor acceleration over time for each control node.
        3. Acceleration time-history for the entire model.
    
        The animation updates the seismic demand for each time step, displaying:
        - A plot of floor displacements, where the color of the line changes based on the maximum drift threshold exceeded.
        - A plot of floor accelerations.
        - A plot of the acceleration time-history.
    
        Parameters:
        ----------
        control_nodes : list
            A list of nodes (floors) in the model, representing the control points for displacement and acceleration.
    
        acc : numpy.ndarray
            A 1D array of acceleration values corresponding to the time-history of seismic excitation.
    
        dts : numpy.ndarray
            A 1D array of time steps (in seconds) for the NRHA analysis.
    
        nrha_disps : numpy.ndarray
            A 2D array of node displacements (in meters) for each time step and control node.
    
        nrha_accels : numpy.ndarray
            A 2D array of node accelerations (in g) for each time step and control node.
    
        drift_thresholds : list
            A list of drift thresholds that define the damage states for the nodes in the model (e.g., no damage, slight damage, etc.).
    
        output_directory : str, optional
            Directory where the animation will be saved. If None, the animation is displayed but not saved.
    
        plot_label : str, optional
            The label for the saved animation file. Default is 'animation'.
    
        Returns:
        --------
        None
            The function generates an animated plot showing seismic demands and optionally saves it as an MP4 file.
    
        Notes:
        -----
        - Currently, obtaining `nrha_disps` (node displacements) and `nrha_accels` (node accelerations) is not straightforward, but this functionality will be available in the next release.
        
        """
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.5])

        # Create subplots
        ax1 = fig.add_subplot(gs[0, 0])  # Floor displacement
        ax2 = fig.add_subplot(gs[0, 1])  # Floor acceleration
        ax3 = fig.add_subplot(gs[1, :])  # Acceleration time-history

        # Initialize lines
        line1, = ax1.plot([], [], color="blue", linewidth=self.line_widths['medium'], marker='o', markersize=self.marker_sizes['small'])
        line2, = ax2.plot([], [], color="red", linewidth=self.line_widths['medium'], marker='o', markersize=self.marker_sizes['small'])
        line3, = ax3.plot([], [], color="green", linewidth=self.line_widths['medium'])

        # Set up subplots
        self._set_plot_style(ax1, title="Floor Displacement (in m)", ylabel='Floor No.')
        self._set_plot_style(ax2, title="Floor Acceleration (in g)", ylabel='Floor No.')
        self._set_plot_style(ax3, title="Acceleration Time-History", xlabel='Time (s)', ylabel='Acceleration (g)')

        ax1.set_ylim(0.0, len(control_nodes))
        ax2.set_ylim(0.0, len(control_nodes))
        ax3.set_xlim(0, dts[-1])
        ax3.set_ylim(np.floor(acc.min()), np.ceil(acc.max()))

        # Add damage state legend
        legend_elements = [Line2D([0], [0], color=c, lw=3, label=state) for c, state in zip(self.colors['damage_states'], ['No Damage', 'Slight Damage', 'Moderate Damage', 'Extensive Damage', 'Complete Damage'])]
        ax1.legend(handles=legend_elements, loc="upper right", fontsize=self.font_sizes['legend'])

        # Animation update function
        def update(frame):
            disp_values = nrha_disps[frame, :]
            accel_values = nrha_accels[frame, :]
            drift_values = np.abs(np.diff(disp_values))

            # Update line data
            line1.set_data(disp_values, range(len(control_nodes)))
            line2.set_data(accel_values, range(len(control_nodes)))
            line3.set_data(dts[:frame], acc[:frame])

            # Update line color based on maximum drift threshold exceeded
            max_drift_threshold_index = np.max(np.where(np.max(drift_values) > drift_thresholds)[0]) if np.any(drift_values > drift_thresholds) else 0
            line1.set_color(self.colors['damage_states'][max_drift_threshold_index])

            return line1, line2, line3

        # Create animation
        ani = FuncAnimation(fig, update, frames=len(dts), interval=1, blit=True, repeat=False)

        # Save animation if output_directory is provided
        if output_directory:
            ani.save(f'{output_directory}/{plot_label}.mp4', writer='ffmpeg', fps=30, dpi=self.resolution)

        plt.tight_layout()
        plt.show()