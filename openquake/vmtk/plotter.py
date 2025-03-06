import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.animation import FuncAnimation

class plotter:
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
        self.font_name = 'Helvetica'

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
        y = [float(node) for node in control_nodes]  # Convert all control nodes to float
        x = [peak_drift_list[i // 2] if i < 2 * (len(control_nodes) - 1) else 0.0
             for i in range(2 * (len(control_nodes) - 1) + 1)]

        return x, y

    def plot_cloud_analysis(self,
                            cloud_dict,
                            output_directory=None,
                            plot_label='cloud_analysis_plot',
                            xlabel='Peak Ground Acceleration, PGA [g]',
                            ylabel=r'Maximum Peak Storey Drift, $\theta_{max}$ [%]'):

        """Plot the cloud analysis results."""
        fig, ax = plt.subplots(figsize=(6, 6))
        self._set_plot_style(ax, xlabel=xlabel, ylabel=ylabel)

        ax.scatter(cloud_dict['imls'], cloud_dict['edps'], color=self.colors['gem'][2], s=self.marker_sizes['medium'], alpha=0.5, label='Cloud Data', zorder=0)
        for i in range(len(cloud_dict['damage_thresholds'])):
            ax.scatter(cloud_dict['medians'][i], cloud_dict['damage_thresholds'][i], color=self.colors['fragility'][i], s=self.marker_sizes['large'], alpha=1.0, zorder=2)

        ax.plot(cloud_dict['fitted_x'], cloud_dict['fitted_y'], linestyle='solid', color=self.colors['gem'][1], lw=self.line_widths['thick'], label='Cloud Regression', zorder=1)
        ax.plot([min(cloud_dict['imls']), max(cloud_dict['imls'])], [cloud_dict['upper_limit'], cloud_dict['upper_limit']], '--', color=self.colors['gem'][-1], label='Upper Censoring Limit')
        ax.plot([min(cloud_dict['imls']), max(cloud_dict['imls'])], [cloud_dict['lower_limit'], cloud_dict['lower_limit']], '-.', color=self.colors['gem'][-1], label='Lower Censoring Limit')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([min(cloud_dict['imls']), max(cloud_dict['imls'])])
        ax.set_ylim([min(cloud_dict['edps']), max(cloud_dict['edps'])])
        ax.legend(fontsize=self.font_sizes['legend'])

        self._save_plot(output_directory, plot_label)

    def plot_fragility_analysis(self,
                                cloud_dict,
                                output_directory=None,
                                plot_label='fragility_plot',
                                xlabel='Peak Ground Acceleration, PGA [g]'):

        """Plot the fragility analysis results."""
        fig, ax = plt.subplots(figsize=(6, 6))
        self._set_plot_style(ax, xlabel=xlabel, ylabel='Probability of Exceedance')

        for i in range(len(cloud_dict['medians'])):
            ax.plot(cloud_dict['intensities'], cloud_dict['poes'][:, i], linestyle='solid', color=self.colors['fragility'][i], lw=self.line_widths['thick'], label=f'DS{i+1}')

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

        """Plot the demand profiles for peak drifts and accelerations."""
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
        """Plot analysis results including cloud analysis, fragility, and demand profiles."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        plt.rcParams['axes.axisbelow'] = True

        # Cloud Analysis
        self._set_plot_style(ax1, xlabel=cloud_xlabel, ylabel=cloud_ylabel)
        ax1.scatter(cloud_dict['imls'], cloud_dict['edps'], color=self.colors['gem'][2], s=self.marker_sizes['medium'], alpha=0.5, label='Cloud Data', zorder=0)
        for i in range(len(cloud_dict['damage_thresholds'])):
            ax1.scatter(cloud_dict['medians'][i], cloud_dict['damage_thresholds'][i], color=self.colors['fragility'][i], s=self.marker_sizes['large'], alpha=1.0, zorder=2)
        ax1.plot(cloud_dict['fitted_x'], cloud_dict['fitted_y'], linestyle='solid', color=self.colors['gem'][1], lw=self.line_widths['thick'], label='Cloud Regression', zorder=1)
        ax1.plot([min(cloud_dict['imls']), max(cloud_dict['imls'])], [cloud_dict['upper_limit'], cloud_dict['upper_limit']], '--', color=self.colors['gem'][-1], label='Upper Censoring Limit')
        ax1.plot([min(cloud_dict['imls']), max(cloud_dict['imls'])], [cloud_dict['lower_limit'], cloud_dict['lower_limit']], '-.', color=self.colors['gem'][-1], label='Lower Censoring Limit')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend(fontsize=self.font_sizes['legend'])

        # Fragility Analysis
        self._set_plot_style(ax2, xlabel=cloud_xlabel, ylabel='Probability of Exceedance')
        for i in range(len(cloud_dict['medians'])):
            ax2.plot(cloud_dict['intensities'], cloud_dict['poes'][:, i], linestyle='solid', color=self.colors['fragility'][i], lw=self.line_widths['thick'], label=f'DS{i+1}')
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
        """Plot the vulnerability analysis results."""
        fig, ax1 = plt.subplots(figsize=(14, 8))
        self._set_plot_style(ax1, xlabel=xlabel, ylabel='Simulated Loss Ratio')

        # Simulate Beta distributions
        simulated_data = []
        intensity_labels = []
        for j, mean_loss in enumerate(loss):
            variance = (cov[j] * mean_loss) ** 2
            alpha = mean_loss * (mean_loss * (1 - mean_loss) / variance - 1)
            beta_param = (1 - mean_loss) * (mean_loss * (1 - mean_loss) / variance - 1)
            data = np.random.beta(alpha, beta_param, 10000)
            simulated_data.append(data)
            intensity_labels.extend([intensities[j]] * len(data))

        # Create DataFrame for seaborn
        df_sns = pd.DataFrame({
            'Intensity Measure': intensity_labels,
            'Simulated Data': np.concatenate(simulated_data)
        })

        # Violin plot
        violin = sns.violinplot(x='Intensity Measure', y='Simulated Data', data=df_sns, scale='width', bw=0.2, inner=None, ax=ax1, zorder=1)
        sns.stripplot(x='Intensity Measure', y='Simulated Data', data=df_sns, color='k', size=1, alpha=0.5, ax=ax1, zorder=3)

        # Loss curve on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(range(len(intensities)), loss, marker='o', linestyle='-', color='blue', label="Loss Curve", zorder=2)
        ax2.set_ylabel(ylabel, fontsize=self.font_sizes['labels'], color='blue', rotation=270, labelpad=20)
        ax2.tick_params(axis='y', labelcolor='blue')

        # Customize x-axis
        ax1.set_xticks(range(len(intensities)))
        ax1.set_xticklabels([f"{x:.3f}" for x in intensities], rotation=45, ha='right', fontsize=self.font_sizes['ticks'])

        # Add legends
        beta_patch = mpatches.Patch(color=violin.collections[0].get_facecolor()[0], label="Beta Distribution")
        ax1.legend(handles=[beta_patch], loc='upper left', fontsize=self.font_sizes['legend'], bbox_to_anchor=(0, 1), ncol=1)
        ax2.legend(loc='upper left', fontsize=self.font_sizes['legend'], bbox_to_anchor=(0, 0.95), ncol=1)

        self._save_plot(output_directory, plot_label)

    def plot_slf_model(self,
                       out,
                       cache,
                       xlabel,
                       output_directory=None,
                       plot_label='slf'):

        """Plot the storey loss function generator output."""
        keys_list = list(cache.keys())
        for i, current_key in enumerate(keys_list):
            rlz = len(cache[current_key]['total_loss_storey'])
            total_loss_storey_array = np.array([cache[current_key]['total_loss_storey'][i] for i in range(rlz)])

            fig, ax = plt.subplots(figsize=(8, 6))
            self._set_plot_style(ax, xlabel=xlabel, ylabel='Storey Loss')

            for i in range(rlz):
                ax.scatter(out[current_key]['edp_range'], total_loss_storey_array[i, :], color=self.colors['gem'][3], s=self.marker_sizes['small'], alpha=0.5)

            ax.fill_between(out[current_key]['edp_range'], cache[current_key]['empirical_16th'], cache[current_key]['empirical_84th'], color='gray', alpha=0.3, label=r'16$^{\text{th}}$-84$^{\text{th}}$ Percentile')
            ax.plot(out[current_key]['edp_range'], cache[current_key]['empirical_median'], lw=self.line_widths['medium'], color='blue', label='Median')
            ax.plot(out[current_key]['edp_range'], out[current_key]['slf'], color='black', lw=self.line_widths['medium'], label='Storey Loss')

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
        """Animate the seismic demands for a single nonlinear time-history analysis run."""
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
