import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter

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

FRAG_COLORS = ['green', 'yellow', 'orange', 'red']  
DS_COLORS = ['blue','green', 'yellow', 'orange', 'red'] # For animation
DS_LABELS = ['No Damage','Slight Damage','Moderate Damage','Extensive Damage','Complete Damage']
GEM_COLORS  = ["#0A4F4E","#0A4F5E","#54D7EB","#54D6EB","#399283","#399264","#399296"]


class plotter():

    def __init__(self):
        pass
    
    def duplicate_for_drift(self,
                            peak_drift_list,
                            control_nodes):
        """
        Creates data to process box plots for peak storey drifts
        -----
        Input
        -----
        :param peak_drift_list:         list          Peak Storey Drifts 
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
            x.extend((peak_drift_list[i],peak_drift_list[i]))
        y.append(float(control_nodes[i+1]))
        x.append(0.0)
        
        return x, y

    
    def plot_cloud_analysis(self,
                            cloud_dict, 
                            output_directory, 
                            plot_label = 'cloud_analysis_plot',
                            xlabel = 'Peak Ground Acceleration, PGA [g]', 
                            ylabel = r'Maximum Peak Storey Drift, $\theta_{max}$ [%]'):
    
        """
        Plots the cloud analysis results 
    
        Parameters
        ----------
        cloud_dict:                    dict                Direct output from do_cloud_analysis function
        output_directory:            string                Output directory path
        plot_label:                  string                Designated filename for plot (default set to "cloud_analysis_plot")
        xlabel:                      string                X-axis label (default set to mpsd)
        ylabel:                      string                Y-axis label (default set to pga)
        
        Returns
        -------
        None.
    
        """
        
        ### Initialise the figure    
        plt.rcParams['figure.figsize'] = [6, 6]
        fig, ax = plt.subplots()
    
        plt.scatter(cloud_dict['imls'], cloud_dict['edps'], color = GEM_COLORS[2], s=MARKER_SIZE_2, alpha = 0.5, label = 'Cloud Data',zorder=0)                   # Plot the cloud scatter 
        for i in range(len(cloud_dict['damage_thresholds'])):
            plt.scatter(cloud_dict['medians'][i], cloud_dict['damage_thresholds'][i], color = FRAG_COLORS[i], s = MARKER_SIZE_1, alpha=1.0, zorder=2)
    
        plt.plot(cloud_dict['fitted_x'], cloud_dict['fitted_y'], linestyle = 'solid', color = GEM_COLORS[1], lw=LINEWIDTH_1, label = 'Cloud Regression', zorder=1) # Plot the regressed fit
    
        plt.plot([min(cloud_dict['imls']), max(cloud_dict['imls'])],[cloud_dict['upper_limit'],cloud_dict['upper_limit']],'--',color=GEM_COLORS[-1], label = 'Upper Censoring Limit') # Plot the upper limit
        plt.plot([min(cloud_dict['imls']), max(cloud_dict['imls'])],[cloud_dict['lower_limit'],cloud_dict['lower_limit']],'-.',color=GEM_COLORS[-1], label = 'Lower Censoring Limit') # Plot the lower limit
    
        plt.xlabel(xlabel, fontsize = FONTSIZE_1, **HFONT)
        plt.ylabel(ylabel, fontsize = FONTSIZE_1, **HFONT)
    
        plt.xticks(fontsize=FONTSIZE_2, rotation=0)
        plt.yticks(fontsize=FONTSIZE_2, rotation=0)
    
        plt.grid(visible=True, which='major')
        plt.grid(visible=True, which='minor')
        
        plt.xscale('log')
        plt.yscale('log')
        
        plt.xlim([min(cloud_dict['imls']), max(cloud_dict['imls'])])
        plt.ylim([min(cloud_dict['edps']), max(cloud_dict['edps'])])
        
        plt.legend()
        plt.savefig(f'{output_directory}/{plot_label}.png', dpi=RESOLUTION, format='png')
        plt.show()



    def plot_demand_profiles(self,
                             peak_drift_list, 
                             peak_accel_list, 
                             control_nodes, 
                             output_directory,
                             plot_label):
        """
        Plots the demand profiles associated with each record of cloud analysis
    
        Parameters
        ----------
        peak_drift_list:               list                Peak storey drifts quantities from analysis
        peak_accel_list:               list                Peak floor acceleration quantities from analysis
        control_nodes:                 list                Nodes of the MDOF system
        output_directory:            string                Output directory path  
        Returns
        -------
        None.
    
        """
               
        ### Initialise the figure
        plt.figure(figsize=(12, 6))
        plt.rcParams['axes.axisbelow'] = True
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)
        
        ### get number of storeys
        nst = len(control_nodes)-1
        
        ### plot the results
        for i in range(len(peak_drift_list)):
            
            x,y = self.duplicate_for_drift(peak_drift_list[i][:,0],control_nodes)
            ax1.plot([float(i)*100 for i in x], y, linewidth=LINEWIDTH_2, linestyle = 'solid', color = GEM_COLORS[1], alpha = 0.7)
            ax1.set_xlabel(r'Peak Storey Drift, $\theta_{max}$ [%]',fontsize = FONTSIZE_2, **HFONT)
            ax1.set_ylabel('Floor No.', fontsize = FONTSIZE_2, **HFONT)
            ax1.grid(visible=True, which='major')
            ax1.grid(visible=True, which='minor')
            ax1.set_yticks(np.linspace(0,nst,nst+1), labels = np.linspace(0,nst,nst+1), minor = False, fontsize=FONTSIZE_3)
            xticks = np.linspace(0,5,11)
            ax1.set_xticks(xticks, labels=xticks, minor=False, fontsize=FONTSIZE_3)
            ax1.set_xlim([0, 5.0])
    
            ax2.plot([float(x)/9.81 for x in peak_accel_list[i][:,0]], control_nodes, linewidth=LINEWIDTH_2, linestyle = 'solid', color = GEM_COLORS[0], alpha=0.7)
            ax2.set_xlabel(r'Peak Floor Acceleration, $a_{max}$ [g]', fontsize = FONTSIZE_2, **HFONT)
            ax2.set_ylabel('Floor No.', fontsize = FONTSIZE_2, **HFONT)
            ax2.grid(visible=True, which='major')
            ax2.grid(visible=True, which='minor')
            ax2.set_yticks(np.linspace(0,nst,nst+1), labels = np.linspace(0,nst,nst+1), minor = False, fontsize=FONTSIZE_3)
            xticks = np.linspace(0,5,11)
            ax2.set_xticks(xticks, labels=xticks, minor=False, fontsize=FONTSIZE_3)
            ax2.set_xlim([0, 5.0])         
    
        plt.savefig(f'{output_directory}/{plot_label}.png', dpi=RESOLUTION, format='png')
        plt.show()

    
    def plot_fragility_analysis(self,
                                cloud_dict, 
                                output_directory, 
                                plot_label = 'fragility_plot',
                                xlabel = 'Peak Ground Acceleration, PGA [g]'):
    
        """
        Plots the cloud analysis results 
    
        Parameters
        ----------
        cloud_dict:                    dict                Direct output from do_cloud_analysis function
        output_directory:            string                Output directory path
        plot_label:                  string                Designated filename for plot (default set to "cloud_analysis_plot")
        xlabel:                      string                X-axis label (default set to pga)
        
        Returns
        -------
        None.
    
        """
        
        ### Plot the cloud    
        plt.rcParams['figure.figsize'] = [6, 6]
        fig, ax = plt.subplots()
    
        for i in range(len(cloud_dict['medians'])):
            plt.plot(cloud_dict['intensities'], cloud_dict['poes'][:,i], linestyle = 'solid', color = FRAG_COLORS[i], lw=LINEWIDTH_1, label = f'DS{i+1}') # Plot the regressed fit
    
        plt.xlabel(xlabel, fontsize = FONTSIZE_1, **HFONT)
        plt.ylabel('Probability of Exceedance', fontsize = FONTSIZE_1, **HFONT)
    
        plt.xticks(fontsize=FONTSIZE_2, rotation=0)
        plt.yticks(fontsize=FONTSIZE_2, rotation=0)
    
        plt.grid(visible=True, which='major')
        plt.grid(visible=True, which='minor')
        plt.xlim([0,5])
        plt.ylim([0,1])
        
        plt.legend()
        plt.savefig(f'{output_directory}/{plot_label}.png', dpi=RESOLUTION, format='png')
        plt.show()


    def plot_ansys_results(self,
                           cloud_dict,
                           peak_drift_list,
                           peak_accel_list,
                           control_nodes,
                           output_directory,
                           plot_label,
                           cloud_xlabel = 'PGA',
                           cloud_ylabel = 'MPSD'):
        
        ### Initialise the figure
        plt.figure(figsize=(10, 10))
        plt.rcParams['axes.axisbelow'] = True
        ax1 = plt.subplot(2,2,1)
        ax2 = plt.subplot(2,2,2)
        ax3 = plt.subplot(2,2,3)
        ax4 = plt.subplot(2,2,4)
            
        # First: Cloud
        ax1.scatter(cloud_dict['imls'], cloud_dict['edps'], color = GEM_COLORS[2], s=MARKER_SIZE_2, alpha = 0.5, label = 'Cloud Data',zorder=0)                   # Plot the cloud scatter 
        for i in range(len(cloud_dict['damage_thresholds'])):
            ax1.scatter(cloud_dict['medians'][i], cloud_dict['damage_thresholds'][i], color = FRAG_COLORS[i], s = MARKER_SIZE_1, alpha=1.0, zorder=2)
        ax1.plot(cloud_dict['fitted_x'], cloud_dict['fitted_y'], linestyle = 'solid', color = GEM_COLORS[1], lw=LINEWIDTH_1, label = 'Cloud Regression', zorder=1) # Plot the regressed fit
        ax1.plot([min(cloud_dict['imls']), max(cloud_dict['imls'])],[cloud_dict['upper_limit'],cloud_dict['upper_limit']],'--',color=GEM_COLORS[-1], label = 'Upper Censoring Limit') # Plot the upper limit
        ax1.plot([min(cloud_dict['imls']), max(cloud_dict['imls'])],[cloud_dict['lower_limit'],cloud_dict['lower_limit']],'-.',color=GEM_COLORS[-1], label = 'Lower Censoring Limit') # Plot the lower limit
        ax1.set_xlabel(cloud_xlabel, fontsize = FONTSIZE_1, **HFONT)
        ax1.set_ylabel(cloud_ylabel, fontsize = FONTSIZE_1, **HFONT)
        ax1.set_xticks(np.linspace(np.log(min(cloud_dict['imls'])),np.log(max(cloud_dict['imls']))), labels = np.linspace(np.log(min(cloud_dict['imls'])),np.log(max(cloud_dict['imls']))), minor = False, fontsize=FONTSIZE_3)
        ax1.set_yticks(np.linspace(np.log(min(cloud_dict['edps'])),np.log(max(cloud_dict['edps']))), labels = np.linspace(np.log(min(cloud_dict['edps'])),np.log(max(cloud_dict['edps']))), minor = False, fontsize=FONTSIZE_3)
        ax1.grid(visible=True, which='major')
        ax1.grid(visible=True, which='minor')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        plt.legend()
            
        # Second: Fragility
        for i in range(len(cloud_dict['medians'])):
            ax2.plot(cloud_dict['intensities'], cloud_dict['poes'][:,i], linestyle = 'solid', color = FRAG_COLORS[i], lw=LINEWIDTH_1, label = f'{DS_LABELS[i+1]}') # Plot the regressed fit
    
        ax2.set_xlabel(cloud_xlabel, fontsize = FONTSIZE_1, **HFONT)
        ax2.set_ylabel('Probability of Exceedance', fontsize = FONTSIZE_1, **HFONT)
    
        ax2.set_xticks(np.linspace(0,5,6), labels = np.round(np.linspace(0,5,6),2), minor = False, fontsize=FONTSIZE_3)
        ax2.set_yticks(np.linspace(0,1,11), labels =np.round(np.linspace(0,1,11),2), minor = False, fontsize=FONTSIZE_3)
    
        ax2.grid(visible=True, which='major')
        ax2.grid(visible=True, which='minor')
        ax2.set_xlim([0,5])
        ax2.set_ylim([0,1])
        ax2.legend()
    
        # Third: Demands
        nst = len(control_nodes)-1    
        for i in range(len(peak_drift_list)):
            x,y = self.duplicate_for_drift(peak_drift_list[i][:,0],control_nodes)
            ax3.plot([float(i)*100 for i in x], y, linewidth=LINEWIDTH_2, linestyle = 'solid', color = GEM_COLORS[1], alpha = 0.7)
            ax3.set_xlabel(r'Peak Storey Drift, $\theta_{max}$ [%]',fontsize = FONTSIZE_2, **HFONT)
            ax3.set_ylabel('Floor No.', fontsize = FONTSIZE_2, **HFONT)
            ax3.grid(visible=True, which='major')
            ax3.grid(visible=True, which='minor')
            ax3.set_yticks(np.linspace(0,nst,nst+1), labels = np.linspace(0,nst,nst+1), minor = False, fontsize=FONTSIZE_3)
            xticks = np.linspace(0,5,11)
            ax3.set_xticks(xticks, labels=xticks, minor=False, fontsize=FONTSIZE_3)
            ax3.set_xlim([0, 5.0])
    
            ax4.plot([float(x) for x in peak_accel_list[i][:,0]], control_nodes, linewidth=LINEWIDTH_2, linestyle = 'solid', color = GEM_COLORS[0], alpha=0.3)
            ax4.set_xlabel(r'Peak Floor Acceleration, $a_{max}$ [g]', fontsize = FONTSIZE_2, **HFONT)
            ax4.set_ylabel('Floor No.', fontsize = FONTSIZE_2, **HFONT)
            ax4.grid(visible=True, which='major')
            ax4.grid(visible=True, which='minor')
            ax4.set_yticks(np.linspace(0,nst,nst+1), labels = np.linspace(0,nst,nst+1), minor = False, fontsize=FONTSIZE_3)
            xticks = np.linspace(0,5,11)
            ax4.set_xticks(xticks, labels=xticks, minor=False, fontsize=FONTSIZE_3)
            ax4.set_xlim([0, 5.0])         
        
        plt.tight_layout()
        plt.savefig(f'{output_directory}/{plot_label}.png', dpi=RESOLUTION, format='png')
        plt.show()

    def plot_multiple_stripe_analysis(msa_dict,
                                      output_directory, 
                                      plot_label = 'multiple_stripe_analysis_plot',
                                      xlabel = r'Maximum Peak Storey Drift, $\theta_{max}$ [%]', 
                                      ylabel = 'Peak Ground Acceleration, PGA [g]'):
        
        """
        Creates a combined subplot of two figures for multiple stripe analysis:
        - First figure: Stripe analysis (IMLs vs EDPs)
        - Second figure: Fitted fragilities (Exceedance probabilities for different thresholds)
    
        Parameters
        ----------
        msa_dict:                      dict                Direct output from do_multiple_stripe_analysis function
        output_directory:            string                Output directory path
        plot_label:                  string                Designated filename for plot (default set to "cloud_analysis_plot")
        xlabel:                      string                X-axis label (default set to mpsd)
        ylabel:                      string                Y-axis label (default set to pga)
        
        Returns
        -------
        None.
    
        """
    
        def plot_stripe_analysis(imls, 
                                 edps, 
                                 damage_thresholds, 
                                 xlabel, 
                                 ylabel, 
                                 ax):
            
            """Plots the stripe analysis (IMLs vs EDPs) on a given axis"""
            for i, threshold in enumerate(damage_thresholds):
                for j, im in enumerate(imls):
                    exceedances = np.sum(edps[j, :] >= threshold)
                    ax.scatter(edps[j, :], [im] * len(edps[j, :]), color = GEM_COLORS[1], s=MARKER_SIZE_2, alpha = 0.5, label = 'MSA Data',zorder=0)
            
            # Add vertical lines for the damage thresholds
            for i, threshold in enumerate(damage_thresholds):
                ax.axvline(x=threshold, color=FRAG_COLORS[i], linestyle='--', label=f'Threshold {threshold}')
    
            ax.set_xlabel(xlabel,fontsize = FONTSIZE_2, **HFONT)
            ax.set_ylabel(ylabel, fontsize = FONTSIZE_2, **HFONT)
            ax.grid(visible=True, which='major')
            ax.grid(visible=True, which='minor')
            ax.set_xlim([0, np.max(edps)])
    
        def plot_exceedance_fit(imls, 
                                num_exc, 
                                num_gmr, 
                                eta, 
                                beta, 
                                threshold, 
                                xlabel,
                                color,
                                ax):
            
            """Plot the exceedance fit for the fragility curve on a given axis"""
            fitted_exceedance = stats.norm.cdf(np.log(imls / eta) / beta)    
            ax.plot(imls, fitted_exceedance, label=f"Fitted Lognormal (Threshold {threshold})", color=color)
            ax.scatter(imls, num_exc / num_gmr, color = color, s=MARKER_SIZE_2, alpha = 0.5, label = 'Observed Exceedances',zorder=0)
            ax.set_xlabel(xlabel, fontsize = FONTSIZE_1, **HFONT)
            ax.set_ylabel('Probability of Exceedance', fontsize = FONTSIZE_1, **HFONT)
            ax.legend()
            ax.grid(visible=True, which='major')
            ax.grid(visible=True, which='minor')
    
    
        # Extract values from msa_dict
        imls = msa_dict['imls']
        edps = msa_dict['edps']
        damage_thresholds = msa_dict['damage_thresholds']
        
        ### Initialise the figure
        plt.figure(figsize=(12, 6))
        plt.rcParams['axes.axisbelow'] = True
        ax1 = plt.subplot(1,2,1)
        ax2 = plt.subplot(1,2,2)
    
        # Plot the stripe analysis on the first axis
        plot_stripe_analysis(imls, 
                             edps, 
                             damage_thresholds, 
                             xlabel, 
                             ylabel, 
                             ax1)
    
        # Loop over all damage thresholds to plot the fragility fits
        for i, threshold in enumerate(damage_thresholds):
            eta = msa_dict['medians'][i]
            beta = msa_dict['betas_total'][i]
            color = FRAG_COLORS[i]
            num_exc = np.array([np.sum(edp >= threshold) for edp in edps])
            num_gmr = np.full(len(imls), len(edps[0]))  # Number of ground motions at each IM level
    
            # Plot the exceedance fit for the current threshold on the second axis
            plot_exceedance_fit(imls, num_exc, num_gmr, eta, beta, threshold, xlabel, color, ax2)
    
        # Adjust layout for better readability
        plt.tight_layout()
        plt.savefig(f'{output_directory}/{plot_label}.png', dpi=RESOLUTION, format='png')        
        plt.show()


    
    def animate_model_run(self,control_nodes, acc, dts, nrha_disps, nrha_accels, drift_thresholds, pflag=True):
        """
        Animates the seismic demands for a single nonlinear time-history analysis run 
        Parameters
        ----------
        control_nodes:                 list                Control nodes of the MDOF system
        acc:                          array                Acceleration values of the applied time-history
        dts:                          array                Pseudo-time values of the applied time-history
        nrha_disps:                   array                Nodal displacement values, output from do_nrha_analysis method
        nrha_accels:                  array                Relative nodal acceleration values, output from do_nrha_analysis method
        drift_thresholds:              list                Drift-based damage thresholds
        
        Returns
        -------
        None.
    
        """
            
        # Set up the figure and the GridSpec layout
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.5])
    
        # Create square subplots for the first row
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
    
        # Create a horizontal subplot that spans the bottom row
        ax3 = fig.add_subplot(gs[1, :])
    
        # Data for the animation
        x1 = nrha_disps
        x2 = nrha_accels
    
        x3 = dts
        y3 = acc
    
        # Initial plots for each subplot
        line1, = ax1.plot([], [], color="blue",  linewidth=LINEWIDTH_2, marker='o', markersize=MARKER_SIZE_3)
        line2, = ax2.plot([], [], color="red",   linewidth=LINEWIDTH_2, marker='o', markersize=MARKER_SIZE_3)
        line3, = ax3.plot([], [], color="green", linewidth=LINEWIDTH_2)
    
        # Set up each subplot
        ax1.set_title("Floor Displacement (in m)", **HFONT)
        ax2.set_title("Floor Acceleration (in g)", **HFONT)
        ax3.set_title("Acceleration Time-History", **HFONT)
        ax1.set_ylim(0.0, len(control_nodes))
        ax2.set_ylim(0.0, len(control_nodes))
        ax3.set_xlim(0, dts[-1])
        ax3.set_ylim(np.floor(acc.min()), np.ceil(acc.max()))
    
        # Set up ticks
        ax1.set_yticks(range(len(control_nodes)))
        ax1.set_yticklabels([f"Floor {i}" for i in range(len(control_nodes))])
    
        ax2.set_yticks(range(len(control_nodes)))
        ax2.set_yticklabels([f"Floor {i}" for i in range(len(control_nodes))])
    
        # --- Enable and customize the grid ---
        # Enable minor ticks for both axes
        ax1.minorticks_on()
        ax2.minorticks_on()
        ax3.minorticks_on()
    
        # Set the major grid locator (spacing of major grid lines)
        ax1.xaxis.set_major_locator(MultipleLocator(1))  # Major grid line every 1 unit on x-axis
        ax1.yaxis.set_major_locator(MultipleLocator(0.5))  # Major grid line every 0.5 unit on y-axis
    
        # Set the minor grid locator (spacing of minor grid lines)
        ax1.xaxis.set_minor_locator(MultipleLocator(0.2))  # Minor grid lines every 0.2 units on x-axis
        ax1.yaxis.set_minor_locator(MultipleLocator(0.1))  # Minor grid lines every 0.1 units on y-axis
    
        # Customize the appearance of the grid lines (major and minor)
        ax1.grid(which='major', color='gray', linestyle='-', linewidth=0.8)
        ax1.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    
        ax2.xaxis.set_major_locator(MultipleLocator(1))  # Major grid line every 1 unit on x-axis
        ax2.yaxis.set_major_locator(MultipleLocator(0.5))  # Major grid line every 0.5 unit on y-axis
        ax2.xaxis.set_minor_locator(MultipleLocator(0.2))  # Minor grid lines every 0.2 units on x-axis
        ax2.yaxis.set_minor_locator(MultipleLocator(0.1))  # Minor grid lines every 0.1 units on y-axis
    
        ax2.grid(which='major', color='gray', linestyle='-', linewidth=0.8)
        ax2.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    
        ax3.xaxis.set_major_locator(MultipleLocator(2))  # Major grid line every 2 units on x-axis
        ax3.yaxis.set_major_locator(MultipleLocator(0.5))  # Major grid line every 0.5 unit on y-axis
        ax3.xaxis.set_minor_locator(MultipleLocator(0.5))  # Minor grid lines every 0.5 units on x-axis
        ax3.yaxis.set_minor_locator(MultipleLocator(0.1))  # Minor grid lines every 0.1 units on y-axis
    
        ax3.grid(which='major', color='gray', linestyle='-', linewidth=0.8)
        ax3.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    
        # Initialize the third line
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
    
        # Add a static legend for damage states in ax1 (floor drift subplot)
        legend_elements = [Line2D([0], [0], color=c, lw=3, label=state) for c, state in zip(DS_COLORS, DS_LABELS)]
        ax1.legend(handles=legend_elements, loc="upper right", fontsize=FONTSIZE_3)
    
        # Initialize tracking variables to remember the maximum threshold exceeded
        max_drift_threshold_index = 0  # Track max threshold index for drift
    
        # Animation update function
        def update(frame):
    
            nonlocal max_drift_threshold_index
            
            # Get current displacements and accelerations for each control node at the current time frame
            disp_values = nrha_disps[frame, :]
            accel_values = nrha_accels[frame, :]
    
            # Calculate drift as the difference in displacement between consecutive floors
            drift_values = np.abs(np.diff(disp_values))  # Absolute drift between consecutive floors
    
            # Determine maximum threshold level exceeded by drift for this frame
            current_drift_threshold_index = max_drift_threshold_index  # Start with the current maximum threshold
    
            for i, threshold in enumerate(drift_thresholds):
                  if np.max(drift_values) > threshold:
                      current_drift_threshold_index = max(current_drift_threshold_index, i)
    
            # Update the maximum drift threshold index reached so far
            max_drift_threshold_index = current_drift_threshold_index
            
            # Set line1 color based on the highest drift threshold reached
            line1.set_color(DS_COLORS[max_drift_threshold_index])
    
            # Update line data
            line1.set_data(disp_values, range(len(control_nodes)))
            line2.set_data(accel_values, range(len(control_nodes)))
            
            # Time-history plot for acceleration data up to the current frame
            line3.set_data(dts[:frame], acc[:frame])
            
            return line1, line2, line3
    
        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(dts), interval=1, blit=True, repeat=False)
    
        # Show the animation
        plt.tight_layout()
        plt.show()  # block=True ensures the animation is displayed in a blocking way
        plt.pause(0.1)
        
        return ani


    def plot_vulnerability_analysis(self, 
                                    intensities,
                                    loss,
                                    cov,
                                    xlabel,
                                    ylabel,
                                    output_directory,
                                    plot_label):
        
        
        # Simulating Beta distributions for each intensity measure
        simulated_data = []
        intensity_labels = []
        
        for j, mean_loss in enumerate(loss):
            variance = (cov[j] * mean_loss) ** 2  # Calculate variance using CoV
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
        ax1.set_ylabel("Simulated Loss Ratio", fontsize=FONTSIZE_1, color='blue')
        ax1.set_xlabel(f"{xlabel}", fontsize=FONTSIZE_1)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.set_ylim(-0.1, 1.2)  # Adjust y-axis range for the violin plot
        
        # Add the legend for the violin plots (Beta distribution)
        # Create a dummy plot handle for the legend, since the violins are not directly plotted as lines
        beta_patch = mpatches.Patch(color=violin.collections[0].get_facecolor()[0], label="Beta Distribution")
        ax1.legend(handles=[beta_patch], loc='upper left', fontsize=FONTSIZE_1, bbox_to_anchor=(0, 1), ncol=1)

        
        # --- Add a second set of x and y axes for the Loss Curve ---
        ax2 = ax1.twinx()  # Create a shared y-axis for the loss curve
        
        # Plot the loss curve on ax2 (now in blue)
        ax2.plot(
            range(len(intensities)), loss, marker='o', linestyle='-', color='blue',
            label="Loss Curve", zorder=2
        )
        
        # Customize the second y-axis (for the loss curve)
        ax2.set_ylabel(f"{ylabel}", fontsize=FONTSIZE_1, color='blue', rotation = 270, labelpad=20)
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim(-0.1, 1.2)  # Adjust y-axis range for the loss curve if needed
        
        # Customize both x-axes to match
        ax1.set_xticks(range(len(intensities)))
        ax1.set_xticklabels([f"{x:.3f}" for x in intensities], rotation=45, ha='right', fontsize= FONTSIZE_3)
                    
        # Add a legend for the loss curve
        ax2.legend(loc='upper left', fontsize=FONTSIZE_1, bbox_to_anchor=(0, 0.95), ncol=1)
        
        # Tight layout and show the combined plot
        plt.tight_layout()
        plt.savefig(f'{output_directory}/{plot_label}.png', dpi=RESOLUTION, format='png')
        plt.show()
