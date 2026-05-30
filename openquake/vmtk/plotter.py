import math
import os

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import platform
if platform.system() == "Windows":
    import openseespy.opensees as ops
elif platform.system() == "Darwin":
    import openseespymac.opensees as ops
else:
    import openseespy.opensees as ops
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import AutoMinorLocator
from scipy import stats
from scipy.interpolate import CubicSpline, interp1d
from scipy.stats import norm


class plotter:
    """
    A class for creating and customizing various types of plots for structural analysis results.

    This class provides methods to visualize data from structural analyses, including cloud
    analysis, fragility analysis, demand profiles, vulnerability analysis, and animations of
    seismic responses.
    It also includes utility methods for setting consistent plot styles and saving plots.

    All static plots are created at a uniform ``figsize`` (default ``(10, 7)``)
    with ``constrained_layout=True`` and are saved **without**
    ``bbox_inches='tight'``.  This guarantees that every exported image has
    exactly the same pixel dimensions (``figsize × resolution``), regardless
    of its content.  Animation panels use ``figsize_anim`` (default
    ``(16, 8)``).  Both attributes can be changed after construction.

    Attributes
    ----------
    figsize : tuple of float
        Width and height in inches for all static (non-animation) plots.
    figsize_anim : tuple of float
        Width and height in inches for animation panels.
    font_sizes : dict
        Dictionary containing font sizes for titles, labels, ticks, and legends.
    line_widths : dict
        Dictionary containing line widths for thick, medium, and thin lines.
    marker_sizes : dict
        Dictionary containing marker sizes for large, medium, and small markers.
    colors : dict
        Dictionary containing color schemes for fragility, damage states, and GEM colors.
    resolution : int
        Resolution for saving plots (default: 400 DPI).
    font_name : str
        Font name for plot text (default: 'Arial').

    Methods
    -------
    _set_plot_style()
        Helper function to set consistent plot style for all plots.

    _save_plot()
        Helper function to save the plot to the specified directory.

    duplicate_for_drift()
        Helper function to create data for box plots of peak storey drifts.

    plot_modes()
        Plots mode shapes

    animate_spo()
        Animates static pushover analyses

    animate_cpo()
        Animates cyclic pushover analyses

    animate_nrha()
        Animates nonlinear time-history analyses

    plot_demand_profiles()
        Plots demand profiles for peak drifts and accelerations from NLTHA output.

    plot_mca_analysis()
        Plots modified cloud analysis results.

    plot_ida_analysis()
        Plots incremental dynamic analysis results.

    plot_msa_analysis()
        Plots multiple stripe analysis results.

    plot_fragility_from_mca()
        Plots fragility analysis results from modified cloud analysis output.

    plot_fragility_from_ida()
        Plots fragility analysis results from incremental dynamic analysis output.

    plot_fragility_from_msa()
        Plots fragility analysis results from multiple stripe analysis output.

    plot_slf_model()
        Plots storey loss function model results.

    plot_vulnerability_function()
        Plots vulnerability analysis results, including Beta distributions and loss curves.

    animate_model_run(control_nodes, acc, dts, nrha_disps, nrha_accels, drift_thresholds,
                      output_directory=None, plot_label='animation')
        Animates the seismic demands for a single nonlinear time-history analysis (NRHA) run.

    """

    def __init__(self):
        """
        Initialize the plotter with default style settings.

        Sets up dictionaries for font sizes, line widths, marker sizes, and color
        schemes used consistently across all plot methods. Also configures the
        default output resolution (DPI), font family, and figure size.

        All figures are created with ``constrained_layout=True`` and saved
        without ``bbox_inches='tight'`` so that every output image has exactly
        the same pixel dimensions (``figsize × resolution``).  Modify
        ``self.figsize`` to change the uniform size of all static plots, or
        ``self.figsize_anim`` for animation panels.

        No parameters are required. All defaults can be overridden by directly
        modifying the instance attributes after construction.

        Example
        -------
        >>> pl = plotter()
        >>> pl.resolution = 300          # lower DPI for faster saves
        >>> pl.font_sizes['title'] = 18  # increase title font size
        >>> pl.figsize = (10, 8)         # change default figure size
        """

        # Define default styles
        self.font_sizes = {
            'title': 18,
            'labels': 16,
            'ticks': 13,
            'legend': 12
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
            'fragility': [
                'green', 'yellow', 'orange', 'red'], 'damage_states': [
                'blue', 'green', 'yellow', 'orange', 'red'], 'gem': [
                "#0A4F4E", "#0A4F5E", "#54D7EB", "#54D6EB", "#399283", "#399264", "#399296"]}
        self.resolution = 400
        self.font_name = 'Arial'
        self.figsize = (10, 7)
        self.figsize_anim = (10, 7)

    def _set_plot_style(
            self,
            ax,
            title=None,
            xlabel=None,
            ylabel=None,
            grid=True):
        """
        Apply a consistent visual style to a Matplotlib axes object.

        Sets the title, axis labels, tick font sizes, and grid visibility using
        the instance-level font and style settings. This is an internal helper
        called by all public plot methods to ensure a uniform appearance.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to style.
        title : str, optional
            Title text to display above the plot. If None, no title is set.
        xlabel : str, optional
            Label for the X-axis. If None, no label is applied.
        ylabel : str, optional
            Label for the Y-axis. If None, no label is applied.
        grid : bool, default True
            If True, enables both major and minor grid lines.

        Returns
        -------
        None
        """
        if title:
            ax.set_title(
                title,
                fontsize=self.font_sizes['title'],
                fontname=self.font_name)
        if xlabel:
            ax.set_xlabel(
                xlabel,
                fontsize=self.font_sizes['labels'],
                fontname=self.font_name)
        if ylabel:
            ax.set_ylabel(
                ylabel,
                fontsize=self.font_sizes['labels'],
                fontname=self.font_name)
        ax.tick_params(axis='both', labelsize=self.font_sizes['ticks'])
        if grid:
            ax.grid(visible=True, which='major')
            ax.grid(visible=True, which='minor')

    def _show(self):
        """
        Display the current figure, suppressing the non-interactive backend
        warning that Matplotlib raises when running in headless environments
        (e.g. CI with the Agg backend).
        """
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*non-interactive.*",
                category=UserWarning,
            )
            plt.show()

    def _save_plot(self, output_directory, plot_label):
        """
        Save the current Matplotlib figure to disk and display it.

        If an output directory is provided, saves the figure as a PNG file at
        the specified resolution.  The figure is saved **without**
        ``bbox_inches='tight'`` so that the on-disk image dimensions match
        ``self.figsize × self.resolution`` exactly.  The plot is always shown
        via ``plt.show()`` after saving (or instead of saving if no directory
        is given).

        Parameters
        ----------
        output_directory : str or None
            Directory where the PNG file will be written. If None, the plot is
            only displayed and not saved.
        plot_label : str
            Filename stem (without extension) for the saved file. The output
            file will be ``<output_directory>/<plot_label>.png``.

        Returns
        -------
        None
        """
        if output_directory:
            plt.savefig(
                f'{output_directory}/{plot_label}.png',
                dpi=self.resolution,
                format='png')
        self._show()

    def duplicate_for_drift(self,
                            peak_drift_list,
                            control_nodes):
        """
        Reshape peak storey drift data into a step-function format for profile plotting.

        For each storey, the drift value is duplicated so that it spans the full height
        of that storey when plotted against elevation. A zero value is appended at the
        roof level to close the profile. This produces the characteristic staircase shape
        used in demand profile visualizations.

        Parameters
        ----------
        peak_drift_list : list or array-like of float
            Peak drift values for each storey (length = number of storeys).
            Index ``i`` corresponds to the drift between ``control_nodes[i]`` and
            ``control_nodes[i+1]``.
        control_nodes : list or array-like
            Elevation (or node tag) values for each floor level, including the ground
            floor. Length must be ``len(peak_drift_list) + 1``.

        Returns
        -------
        x : list of float
            Drift values expanded into step-function format. Each storey drift is
            repeated twice, and a trailing ``0.0`` is appended at the top.
        y : list of float
            Corresponding floor elevation values, also expanded to match ``x``.
            Each storey boundary elevation is repeated twice.
        """
        x = []
        y = []
        for i in range(len(control_nodes) - 1):
            y.extend((float(control_nodes[i]), float(control_nodes[i + 1])))
            x.extend((peak_drift_list[i], peak_drift_list[i]))
        y.append(float(control_nodes[i + 1]))
        x.append(0.0)

        return x, y

    # PLOT MODAL ANALYSIS OUTPUT

    def plot_modes(self,
                   node_list,
                   mode_shape_vectors,
                   T,
                   export_path=None):
        """
        Plots 2-D mode shape profiles in a square grid layout (2x2, 3x3 etc).

        Each mode occupies one cell with two side-by-side profile plots:
        left = X-displacement vs Z (blue), right = Y-displacement vs Z (green).
        Normalised displacement values are annotated next to every node dot.
        A grey undeformed reference line (x=0) is drawn in every panel.

        Sign convention
        ---------------
        Eigenvectors have arbitrary sign.  The method flips each mode so that
        the top-node displacement in the dominant horizontal direction is always
        positive.

        Grid
        ----
        ncols = ceil(sqrt(N)),  nrows = ceil(N / ncols).
        Unused cells in the last row are hidden.

        Parameters
        ----------
        node_list : list of int
            Ordered OpenSees node tags (base node first).
        mode_shape_vectors : list of numpy.ndarray, shape (n_nodes, 3)
            One array per mode; columns are [ux, uy, uz], pre-normalised by
            max abs value as returned by do_modal_analysis.
        T : list of float
            Natural periods [s] for each mode.
        export_path : str, optional
            File path to save the figure.  If None the figure is displayed.

        Returns
        -------
        None
        """
        COL_BASE = '#B71C1C'
        COL_NODE = '#1565C0'
        COL_UNDEF = '#90A4AE'
        COL_ANN = '#37474F'
        COL_GRID = '#EBEBEB'
        COL_X = '#1565C0'
        COL_Y = '#2E7D32'
        BG = 'white'

        node_z = np.array([ops.nodeCoord(tag, 3) for tag in node_list])
        n_nodes = len(node_list)
        num_modes = len(T)
        unique_z = np.unique(node_z)
        z_min, z_max = unique_z[0], unique_z[-1]

        def _fix_sign(phi):
            top = phi[-1, :]
            dom = int(np.argmax(np.abs(top[:2])))
            return -phi if top[dom] < 0 else phi.copy()

        norms = [_fix_sign(np.asarray(mv)) for mv in mode_shape_vectors]

        ncols = math.ceil(math.sqrt(num_modes))
        nrows = math.ceil(num_modes / ncols)

        fig = plt.figure(figsize=(ncols * 5.0, nrows * 4.5), facecolor=BG)
        fig.patch.set_facecolor(BG)
        gs = gridspec.GridSpec(nrows, ncols * 2, figure=fig,
                               hspace=0.50, wspace=0.35,
                               left=0.07, right=0.97,
                               top=0.88, bottom=0.08)

        interp_kind = ('cubic' if len(unique_z) >= 4 else
                       'quadratic' if len(unique_z) == 3 else 'linear')
        z_sm = np.linspace(z_min, z_max, 300)

        for idx in range(ncols * nrows):
            row = idx // ncols
            col = idx % ncols
            gc_x = col * 2
            gc_y = col * 2 + 1

            if idx >= num_modes:
                for gc in (gc_x, gc_y):
                    ax = fig.add_subplot(gs[row, gc])
                    ax.set_visible(False)
                continue

            phi = norms[idx]
            ux = phi[:, 0]
            uy = phi[:, 1]
            ux_sm = interp1d(unique_z, ux, kind=interp_kind)(z_sm)
            uy_sm = interp1d(unique_z, uy, kind=interp_kind)(z_sm)
            xlim_x = max(np.max(np.abs(ux)) * 1.55, 0.12)
            xlim_y = max(np.max(np.abs(uy)) * 1.55, 0.12)
            dom = 'X' if np.max(np.abs(ux)) >= np.max(np.abs(uy)) else 'Y'
            t_base = (f'Mode {idx + 1} [{dom}-dir]  \u2014  '
                      f'$T_{{{idx + 1}}} = {T[idx]:.3f}$ s')

            def _draw(ax, disps, disps_sm, col_line, col_node,
                      xlabel, xlim, title):
                ax.set_facecolor(BG)
                ax.grid(True, color=COL_GRID, lw=0.6, zorder=0)
                ax.set_axisbelow(True)
                ax.axvline(0, color=COL_UNDEF, lw=1.2, ls='-', zorder=1)
                ax.plot(disps_sm, z_sm, color=col_line, lw=2.0,
                        zorder=3, solid_capstyle='round')
                ann_off = xlim * 0.05
                for i in range(n_nodes):
                    z_i = node_z[i]
                    d_i = disps[i]
                    nc = COL_BASE if i == 0 else col_node
                    ax.scatter(d_i, z_i,
                               marker='s' if i == 0 else 'o',
                               s=55 if i == 0 else 40,
                               color=nc, edgecolors='white',
                               linewidths=0.8, zorder=5)
                    ax.text(d_i + ann_off, z_i, f'{d_i:+.3f}',
                            fontsize=6.5, color=COL_ANN,
                            va='center', ha='left', zorder=6)
                ax.set_xlim(-xlim, xlim)
                ax.set_ylim(z_min - 0.4, z_max + 0.4)
                ax.set_xlabel(xlabel, fontsize=8, color=COL_ANN, labelpad=3)
                ax.tick_params(labelsize=7, colors=COL_ANN)
                for sp in ('top', 'right'):
                    ax.spines[sp].set_visible(False)
                ax.spines['left'].set_color(COL_ANN)
                ax.spines['bottom'].set_color(COL_ANN)
                ax.set_title(title, fontsize=8, fontweight='bold',
                             color='#1A237E', pad=7)

            ax_x = fig.add_subplot(gs[row, gc_x])
            ax_x.set_ylabel('Z [m]', fontsize=8, color=COL_ANN, labelpad=3)
            _draw(ax_x, ux, ux_sm, COL_X, COL_NODE,
                  r'$u_x$ (norm.)', xlim_x, t_base)

            ax_y = fig.add_subplot(gs[row, gc_y], sharey=ax_x)
            ax_y.tick_params(labelleft=False)
            _draw(ax_y, uy, uy_sm, COL_Y, '#2E7D32',
                  r'$u_y$ (norm.)', xlim_y, t_base)

        fig.suptitle('OpenSees  —  Modal Analysis  |  Mode Shapes',
                     fontsize=13, fontweight='bold', color='#1A237E')

        if export_path:
            plt.savefig(export_path, dpi=self.resolution,
                        bbox_inches='tight', facecolor=BG)
            self._show()
        else:
            self._show()
        plt.close(fig)

    # ANIMATE STATIC PUSHOVER ANALYSES

    def animate_spo(self,
                    spo_top_disp,
                    spo_rxn,
                    spo_disps,
                    spo_midr,
                    nodeList,
                    elementList,
                    push_dir,
                    phi,
                    export_path,
                    frame_step=5,
                    dpi=100):
        """
        Generate and save an animation of a static (monotonic) pushover analysis.

        Layout — self.figsize_anim
        --------------------------
        Left  (wide) - 2-D deformed model shape with orange load arrows.
        Top-right    - Pushover curve (base shear vs. roof displacement).
        Bottom-right - Base shear vs. maximum inter-storey drift ratio.

        Parameters
        ----------
        spo_top_disp : array-like
        spo_rxn : array-like
        spo_disps : array-like, shape (n_steps, n_floors)
        spo_midr : array-like
        nodeList : list of int
        elementList : list of int
        push_dir : int  (1=X, 2=Y, 3=Z)
        phi : list of float  (floor load pattern, no base node)
        export_path : str
        frame_step : int, optional
            Render every N-th step to reduce frame count and speed up export.
            Default 5.
        dpi : int, optional
            Export resolution. Default 100.
        """
        spo_rxn = np.asarray(spo_rxn)
        spo_midr = np.asarray(spo_midr)
        spo_disps = np.asarray(spo_disps)
        phi_arr = np.asarray(phi, dtype=float)
        phi_norm = phi_arr / phi_arr.max()

        total_steps = len(spo_top_disp)
        frame_indices = np.arange(0, total_steps, frame_step)
        num_frames = len(frame_indices)
        deform_factor = 1

        NodeCoordListX_und = [ops.nodeCoord(tag, 1) for tag in nodeList]
        NodeCoordListY_und = [ops.nodeCoord(tag, 2) for tag in nodeList]
        NodeCoordListZ_und = [ops.nodeCoord(tag, 3) for tag in nodeList]

        if push_dir == 2:
            horiz_und, x_label_model = NodeCoordListY_und, 'Y-Direction [m]'
        elif push_dir == 3:
            horiz_und, x_label_model = NodeCoordListZ_und, 'Z-Direction [m]'
        else:
            horiz_und, x_label_model = NodeCoordListX_und, 'X-Direction [m]'
        vert_und = NodeCoordListZ_und

        max_disp_ever = np.max(np.abs(spo_disps))
        half_x = max(max_disp_ever * 1.5, 0.01)
        arrow_scale = (half_x * 0.40) / spo_rxn.max()
        model_xlim = (
            -(half_x * 0.25 + spo_rxn.max() * arrow_scale * 1.1),
            half_x * 1.25
        )
        model_ylim = (0, max(vert_und) * 1.5)

        # Pre-build element pairs for speed
        ele_pairs = []
        for eleTag in elementList:
            try:
                ni, nj = ops.eleNodes(eleTag)
                ele_pairs.append((nodeList.index(ni), nodeList.index(nj)))
            except Exception:
                continue

        _FS = 11  # uniform fontsize for all SPO animation text

        fig = plt.figure(figsize=self.figsize_anim)
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               width_ratios=[1.1, 1],
                               left=0.07, right=0.97,
                               top=0.95, bottom=0.08,
                               hspace=0.45, wspace=0.35)
        ax_model = fig.add_subplot(gs[:, 0])
        ax_curve = fig.add_subplot(gs[0, 1])
        ax_drift = fig.add_subplot(gs[1, 1])

        # Static background
        ax_model.scatter(horiz_und, vert_und,
                         marker='o', s=40, color='gray', alpha=0.5, zorder=3)
        for ii, jj in ele_pairs:
            ax_model.plot([horiz_und[ii], horiz_und[jj]],
                          [vert_und[ii], vert_und[jj]],
                          color='gray', ls='--', lw=1.0, alpha=0.5, zorder=2)
        ax_model.set_xlabel(x_label_model, fontsize=_FS)
        ax_model.set_ylabel('Elevation [m]', fontsize=_FS)
        ax_model.set_xlim(model_xlim)
        ax_model.set_ylim(model_ylim)
        ax_model.grid(True, ls=':', alpha=0.4)
        ax_model.tick_params(labelsize=_FS)

        n_static_lines = len(ax_model.lines)
        n_static_colls = len(ax_model.collections)

        ax_curve.plot(spo_top_disp, spo_rxn, color='gray', lw=2, alpha=0.5)
        curve_anim, = ax_curve.plot([], [], 'blue', lw=2)
        ax_curve.set_xlabel('Roof Displacement [m]', fontsize=_FS)
        ax_curve.set_ylabel('Base Shear [kN]', fontsize=_FS)
        ax_curve.set_title(
            'Base Shear vs Roof Displacement',
            fontsize=_FS,
            fontweight='bold')
        ax_curve.set_xlim(0, np.max(spo_top_disp) * 1.1)
        ax_curve.set_ylim(0, np.max(spo_rxn) * 1.15)
        ax_curve.grid(True, ls=':', alpha=0.4)
        ax_curve.tick_params(labelsize=_FS)

        ax_drift.plot(spo_midr, spo_rxn, color='gray', lw=2, alpha=0.5)
        drift_anim, = ax_drift.plot([], [], 'green', lw=2)
        ax_drift.set_xlabel('Max ISDR [%]', fontsize=_FS)
        ax_drift.set_ylabel('Base Shear [kN]', fontsize=_FS)
        ax_drift.set_title(
            'Base Shear vs Max ISDR',
            fontsize=_FS,
            fontweight='bold')
        ax_drift.set_xlim(0, np.max(spo_midr) * 1.2)
        ax_drift.set_ylim(0, np.max(spo_rxn) * 1.15)
        ax_drift.grid(True, ls=':', alpha=0.4)
        ax_drift.tick_params(labelsize=_FS)

        def update(anim_frame):
            frame = int(frame_indices[anim_frame])
            while len(ax_model.lines) > n_static_lines:
                ax_model.lines[-1].remove()
            while len(ax_model.collections) > n_static_colls:
                ax_model.collections[-1].remove()

            full_disps = np.concatenate(([0.0], spo_disps[frame]))
            xd = [
                horiz_und[i] +
                full_disps[i] *
                deform_factor for i in range(
                    len(nodeList))]
            zd = list(vert_und)

            ax_model.scatter(
                xd,
                zd,
                marker='o',
                s=50,
                color='#1565C0',
                zorder=5)
            for ii, jj in ele_pairs:
                ax_model.plot([xd[ii], xd[jj]], [zd[ii], zd[jj]],
                              color='#1565C0', lw=2.0, zorder=4)

            current_shear = spo_rxn[frame]
            for i, (xdi, zdi) in enumerate(zip(xd[1:], zd[1:]), start=0):
                arrow_len = phi_norm[i] * current_shear * arrow_scale
                ax_model.annotate(
                    '',
                    xy=(xdi, zdi),
                    xytext=(xdi - arrow_len, zdi),
                    arrowprops=dict(arrowstyle='->', color='#E65100',
                                    lw=1.5, mutation_scale=10),
                    zorder=6
                )

            ax_model.set_title(
                f'Deformed Shape - Step {frame + 1}/{total_steps}',
                fontsize=_FS, fontweight='bold'
            )
            curve_anim.set_data(spo_top_disp[:frame + 1], spo_rxn[:frame + 1])
            drift_anim.set_data(spo_midr[:frame + 1], spo_rxn[:frame + 1])
            return curve_anim, drift_anim

        ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                      interval=50, blit=False)
        if export_path:
            os.makedirs(os.path.dirname(export_path) or '.', exist_ok=True)
            print(
                f'Saving SPO animation ({num_frames} frames) to: {export_path}')
            try:
                if export_path.lower().endswith('.gif'):
                    ani.save(export_path, writer='pillow', dpi=dpi)
                elif export_path.lower().endswith('.mp4'):
                    ani.save(export_path, writer='ffmpeg', dpi=dpi)
                else:
                    ani.save(export_path + '.gif', writer='pillow', dpi=dpi)
            except Exception as e:
                print(f'Failed to save SPO animation: {e}')
        plt.close(fig)

    def animate_cpo(self,
                    cpo_dict,
                    nodeList,
                    elementList,
                    push_dir,
                    export_path):
        """
        Generates and saves the CPO animation using FuncAnimation, showing:
        1. Deformed model shape.
        2. Base shear vs. top displacement (hysteretic curve).
        3. Base shear vs. maximum interstorey drift (hysteretic curve).

        The animation figure uses ``self.figsize_anim`` with
        ``constrained_layout`` so that every exported frame has identical,
        deterministic pixel dimensions.

        Parameters
        ----------
        cpo_dict : dict
            The analysis results dictionary returned by do_cpo_analysis.
        nodeList : list
            List of node tags in the model.
        elementList : list
            List of element tags in the model.
        push_dir : int
            Direction of the pushover analysis (1=X, 2=Y, 3=Z).
        export_path : str
            File path to save the animation (.gif or .mp4).

        Returns
        -------
        None
            The animation is written to ``export_path``; the figure is closed
            afterwards to free memory.
        """
        # Data Extraction and Processing
        cpo_top_disp = cpo_dict['cpo_top_disp']
        cpo_rxn = cpo_dict['cpo_rxn']
        cpo_disps = cpo_dict['cpo_disps']
        cpo_drifts = cpo_dict['cpo_idr']

        deform_factor = 1.0
        total_steps = len(cpo_top_disp)
        max_frames_cpo = 150
        frame_step_cpo = max(1, total_steps // max_frames_cpo)
        frame_indices_cpo = np.arange(0, total_steps, frame_step_cpo)
        if frame_indices_cpo[-1] != total_steps - 1:
            frame_indices_cpo = np.append(frame_indices_cpo, total_steps - 1)
        # Find the drift (with sign) of the floor with the maximum absolute
        # drift at each step.
        max_drift_indices = np.argmax(np.abs(cpo_drifts), axis=1)
        governing_drift_history = cpo_drifts[np.arange(
            total_steps), max_drift_indices]

        # Max absolute drift for setting limits
        max_drift_limit = np.max(np.abs(governing_drift_history))

        # Pre-compute cumulative dissipated energy [kN·m] via trapezoidal rule:
        # dE[i] = 0.5 * (F[i] + F[i-1]) * (u[i] - u[i-1])
        # Energy is always accumulated (absolute value of increment) so the
        # curve is monotonically increasing.
        cpo_top_disp_arr = np.asarray(cpo_top_disp)
        cpo_rxn_arr = np.asarray(cpo_rxn)
        du = np.diff(cpo_top_disp_arr)
        f_avg = 0.5 * (cpo_rxn_arr[:-1] + cpo_rxn_arr[1:])
        dE = np.abs(f_avg * du)          # always positive increment
        cumul_energy = np.concatenate(
            ([0.0], np.cumsum(dE)))  # length = num_frames
        max_energy = cumul_energy[-1] * 1.1

        # Get undeformed coordinates once
        NodeCoordListX_und = [ops.nodeCoord(tag, 1) for tag in nodeList]
        NodeCoordListY_und = [ops.nodeCoord(tag, 2) for tag in nodeList]
        NodeCoordListZ_und = [ops.nodeCoord(tag, 3) for tag in nodeList]

        if push_dir == 1:
            plot_coords_und = (NodeCoordListX_und, NodeCoordListZ_und)
            x_label_model = 'X-Direction [m]'
            y_label_model = 'Z-Direction [m]'
        elif push_dir == 2:
            plot_coords_und = (NodeCoordListY_und, NodeCoordListZ_und)
            x_label_model = 'Y-Direction [m]'
            y_label_model = 'Z-Direction [m]'
        elif push_dir == 3:
            plot_coords_und = (NodeCoordListZ_und, NodeCoordListX_und)
            x_label_model = 'Z-Direction [m]'
            y_label_model = 'X-Direction [m]'
        else:
            plot_coords_und = (NodeCoordListX_und, NodeCoordListZ_und)
            x_label_model = 'X-Direction [m]'
            y_label_model = 'Z-Direction [m]'

        max_abs_coord_x = np.max(np.abs(plot_coords_und[0]))
        max_abs_coord_y = np.max(np.abs(plot_coords_und[1]))
        # For stick models all nodes share x=0; use the max expected deformation
        # (last cpo_top_disp value) to set a sensible x-axis width.
        x_half = max(max_abs_coord_x * 3.0,
                     np.max(np.abs(cpo_top_disp)) * 2.0, 0.01)
        model_x_lim = (-x_half, x_half)
        model_y_lim = (0, max_abs_coord_y * 1.5)

        # Initialize the Figure and Subplots
        fig = plt.figure(figsize=self.figsize_anim)
        gs = gridspec.GridSpec(3, 2, figure=fig,
                               left=0.07, right=0.97,
                               top=0.95, bottom=0.08,
                               hspace=0.55, wspace=0.35)
        ax_model = fig.add_subplot(gs[:, 0])   # full-height left panel
        ax_curve = fig.add_subplot(gs[0, 1])
        ax_drift = fig.add_subplot(gs[1, 1])
        ax_energy = fig.add_subplot(gs[2, 1])

        # Store count of static artists for cleanup in update()
        num_static_lines = len(elementList)
        num_static_collections = 1

        # Static (undeformed) background
        ax_model.scatter(plot_coords_und[0], plot_coords_und[1],
                         marker='o', s=50, color='gray', alpha=0.5,
                         label='Undeformed Nodes')
        for eleTag in elementList:
            try:
                [NodeItag, NodeJtag] = ops.eleNodes(eleTag)
                i = nodeList.index(NodeItag)
                j = nodeList.index(NodeJtag)
            except Exception:
                continue
            x_und = [plot_coords_und[0][i], plot_coords_und[0][j]]
            y_und = [plot_coords_und[1][i], plot_coords_und[1][j]]
            ax_model.plot(
                x_und,
                y_und,
                color='gray',
                linestyle='--',
                linewidth=0.5,
                alpha=0.5)

        _FS = 11  # uniform fontsize for all CPO animation text

        ax_model.set_xlabel(x_label_model, fontsize=_FS)
        ax_model.set_ylabel(y_label_model, fontsize=_FS)
        ax_model.set_title(
            'Deformed Model Shape (Cyclic Pushover)',
            fontsize=_FS,
            fontweight='bold')
        ax_model.set_xlim(model_x_lim)
        ax_model.set_ylim(model_y_lim)
        ax_model.grid(True)
        ax_model.tick_params(labelsize=_FS)

        # Hysteretic Curve (Base Shear vs Top Disp)
        ax_curve.set_xlabel('Top Displacement [m]', fontsize=_FS)
        ax_curve.set_ylabel('Base Shear [kN]', fontsize=_FS)
        ax_curve.set_title('Hysteretic Curve', fontsize=_FS, fontweight='bold')
        ax_curve.plot(cpo_top_disp, cpo_rxn, 'gray', linewidth=2,
                      alpha=0.5, label='History')
        curve_anim, = ax_curve.plot([], [], 'blue', linewidth=2,
                                    label='Current Step')
        ax_curve.legend(loc='lower right', fontsize=_FS)
        max_x_curve = np.max(np.abs(cpo_top_disp)) * 1.1
        max_y_curve = np.max(np.abs(cpo_rxn)) * 1.1
        ax_curve.set_xlim(-max_x_curve, max_x_curve)
        ax_curve.set_ylim(-max_y_curve, max_y_curve)
        ax_curve.grid(True)
        ax_curve.tick_params(labelsize=_FS)

        # Governing Drift Hysteresis (Base Shear vs MIDR)
        ax_drift.set_xlabel('Maximum Interstorey Drift [-]', fontsize=_FS)
        ax_drift.set_ylabel('Base Shear [kN]', fontsize=_FS)
        ax_drift.set_title('Hysteretic Curve', fontsize=_FS, fontweight='bold')
        ax_drift.plot(governing_drift_history, cpo_rxn, 'gray', linewidth=2,
                      alpha=0.5, label='History')
        drift_anim, = ax_drift.plot([], [], 'green', linewidth=2,
                                    label='Current Step')
        ax_drift.legend(loc='lower right', fontsize=_FS)
        ax_drift.set_xlim(-max_drift_limit * 1.1, max_drift_limit * 1.1)
        ax_drift.set_ylim(-max_y_curve, max_y_curve)
        ax_drift.grid(True)
        ax_drift.tick_params(labelsize=_FS)

        # Dissipated Energy vs Step
        ax_energy.set_xlabel('Step [-]', fontsize=_FS)
        ax_energy.set_ylabel('Dissipated Energy [kN\u00b7m]', fontsize=_FS)
        ax_energy.set_title(
            'Cumulative Dissipated Energy',
            fontsize=_FS,
            fontweight='bold')
        ax_energy.plot(
            np.arange(total_steps),
            cumul_energy,
            'gray',
            linewidth=1,
            alpha=0.5,
            label='History')
        energy_anim, = ax_energy.plot([], [], 'green', linewidth=2,
                                      label='Current Step')
        ax_energy.legend(loc='lower right', fontsize=_FS)
        ax_energy.set_xlim(0, total_steps * 1.05)
        ax_energy.set_ylim(0, max_energy if max_energy > 0 else 1.0)
        ax_energy.grid(True)
        ax_energy.tick_params(labelsize=_FS)

        def update(frame):
            # Remove deformed artists from previous frame
            while len(ax_model.lines) > num_static_lines:
                ax_model.lines[-1].remove()
            while len(ax_model.collections) > num_static_collections:
                ax_model.collections[-1].remove()

            # Deformed coordinates
            current_disps_floor = cpo_disps[frame]
            full_node_disps = np.insert(current_disps_floor, 0, 0, axis=0)

            if push_dir == 1:
                X_def = [
                    plot_coords_und[0][i] +
                    full_node_disps[i] *
                    deform_factor for i in range(
                        len(nodeList))]
                Z_def = [plot_coords_und[1][i] for i in range(len(nodeList))]
                plot_coords_def = (X_def, Z_def)
            elif push_dir == 2:
                Y_def = [
                    plot_coords_und[0][i] +
                    full_node_disps[i] *
                    deform_factor for i in range(
                        len(nodeList))]
                Z_def = [plot_coords_und[1][i] for i in range(len(nodeList))]
                plot_coords_def = (Y_def, Z_def)
            elif push_dir == 3:
                Z_def = [
                    plot_coords_und[0][i] +
                    full_node_disps[i] *
                    deform_factor for i in range(
                        len(nodeList))]
                X_def = [plot_coords_und[1][i] for i in range(len(nodeList))]
                plot_coords_def = (Z_def, X_def)
            else:
                plot_coords_def = plot_coords_und

            # Deformed shape
            ax_model.scatter(plot_coords_def[0], plot_coords_def[1],
                             marker='o', s=50, color='blue',
                             label='Deformed Nodes')
            for eleTag in elementList:
                try:
                    [NodeItag, NodeJtag] = ops.eleNodes(eleTag)
                    i = nodeList.index(NodeItag)
                    j = nodeList.index(NodeJtag)
                except Exception:
                    continue
                x_def = [plot_coords_def[0][i], plot_coords_def[0][j]]
                y_def = [plot_coords_def[1][i], plot_coords_def[1][j]]
                ax_model.plot(x_def, y_def, color='blue', linewidth=1.5)

            ax_model.set_title(
                f'Step: {frame}/{total_steps - 1} (Scale: {deform_factor}x)',
                fontsize=_FS,
                fontweight='bold')

            curve_anim.set_data(cpo_top_disp[:frame + 1], cpo_rxn[:frame + 1])
            drift_anim.set_data(governing_drift_history[:frame + 1],
                                cpo_rxn[:frame + 1])
            energy_anim.set_data(np.arange(frame + 1),
                                 cumul_energy[:frame + 1])
            return curve_anim, drift_anim, energy_anim

        ani = animation.FuncAnimation(fig, update,
                                      frames=frame_indices_cpo,
                                      interval=50, blit=False)

        if export_path:
            directory = os.path.dirname(export_path)
            if directory and not os.path.exists(directory):
                print(f"Creating directory: {directory}")
                os.makedirs(directory, exist_ok=True)

        print(f"\nSaving animation to: {export_path}")
        try:
            if export_path.lower().endswith('.gif'):
                ani.save(export_path, writer='pillow', dpi=150)
            elif export_path.lower().endswith('.mp4'):
                ani.save(export_path, writer='ffmpeg', dpi=200)
            else:
                print("WARNING: Animation path extension not recognized. Saving as GIF.")
                ani.save(export_path + ".gif", writer='pillow', dpi=150)
        except Exception as e:
            print(f"Failed to save animation: {e}")

        plt.close(fig)

    # ANIMATE NONLINEAR TIME-HISTORY ANALYSES

    def animate_nrha(self,
                     control_nodes,
                     acc,
                     dts,
                     nrha_disps,
                     nrha_accels,
                     drift_thresholds=None,
                     export_path=None,
                     frame_step=5,
                     dpi=100,
                     collapse_time=None,
                     true_peak_drift=None,
                     true_peak_accel=None):
        """
        Animate the seismic response for a nonlinear time-history analysis (NRHA).

        Four-panel layout (self.figsize_anim)
        -------------------------------------
        Panel 1 - Floor displacement profile [m] vs. elevation.
        Panel 2 - Storey drift profile [%] vs. elevation (staircase style, matching plot_demand_profiles).
        Panel 3 - Floor acceleration profile [g] vs. elevation.
        Panel 4 - Input ground motion time-history with elapsed portion highlighted. If collapse_time is provided, an 'X' marker is drawn at that instant to indicate when the MinMax material limit was exceeded.

        Line colours update cumulatively based on worst damage state reached so
        far (blue -> green -> yellow -> orange -> red) when drift_thresholds given.

        Parameters
        ----------
        control_nodes : array-like of int
        acc : array-like of float  [g]
        dts : array-like of float  [s]
        nrha_disps : ndarray (n_steps, n_nodes)  [m]
        nrha_accels : ndarray (n_steps, n_nodes)  [m/s^2]
        drift_thresholds : list of float or None
        export_path : str or None
        frame_step : int, optional
            Render every N-th timestep. Default 5.
        dpi : int, optional
            Export resolution. Default 100.
        collapse_time : float or None, optional
            Time [s] at which the MinMax material limit was exceeded (i.e. the
            last recorded timestep when conv_index turned -1).  When provided,
            a red 'X' marker is drawn on the ground-motion panel at that instant.
        true_peak_drift : ndarray (n_storeys,) or None, optional
            True peak IDR per storey [ratio] from the full-resolution time-step
            loop (peak_drift[:,0] from do_nrha_analysis).  When provided the
            peak drift annotation shows this value instead of the subsampled max.
        true_peak_accel : ndarray (n_nodes,) or None, optional
            True peak absolute floor acceleration per node [g] from the full-
            resolution loop (peak_accel[:,0] from do_nrha_analysis).  When
            provided the peak accel annotation shows this value instead of the
            subsampled max.

        Returns
        -------
        ani : FuncAnimation
        """
        acc = np.asarray(acc)
        dts = np.asarray(dts)
        nrha_disps = np.asarray(nrha_disps)
        nrha_accels = np.asarray(nrha_accels)

        # Subsample for speed
        frame_indices = np.arange(0, len(dts), frame_step)
        num_frames = len(frame_indices)

        # Storey geometry
        node_z_coords = np.array([ops.nodeCoord(n, 3) for n in control_nodes])
        sorted_idx = np.argsort(node_z_coords)
        control_nodes = np.array(control_nodes)[sorted_idx]
        node_z_coords = node_z_coords[sorted_idx]
        storey_heights = np.diff(node_z_coords)
        n_storeys = len(storey_heights)

        if np.any(storey_heights <= 1e-6):
            print("Warning: Zero or near-zero storey height detected")

        # Pre-compute axis limits from full data for stable axes
        max_abs_disp = np.max(np.abs(nrha_disps)) if nrha_disps.size else 0.01
        max_abs_accel = np.max(
            np.abs(
                nrha_accels /
                9.81)) if nrha_accels.size else 1.0

        # Pre-compute storey drifts for ALL frames to get drift xlim.
        # If collapse_time is given, exclude the final frame (which is at/past
        # the MinMax limit and would produce an artificially large drift
        # value).
        all_drifts_pct = (np.abs(np.diff(nrha_disps, axis=1)) /
                          storey_heights[np.newaxis, :] * 100.0)
        if collapse_time is not None and all_drifts_pct.shape[0] > 1:
            # exclude last (collapsed) step
            drifts_for_xlim = all_drifts_pct[:-1]
        else:
            drifts_for_xlim = all_drifts_pct
        max_drift_pct = np.max(
            drifts_for_xlim) if drifts_for_xlim.size else 1.0
        xlim_drift = max(max_drift_pct * 1.2, 0.5)

        elev_pad = 0.1 * storey_heights.mean()
        ylim_elev = (node_z_coords[0] - elev_pad, node_z_coords[-1] + elev_pad)

        # Figure — 4 rows, height ratios give more space to profiles, less to
        # GM
        fig = plt.figure(figsize=self.figsize_anim, constrained_layout=True)
        gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 0.7], figure=fig)

        ax_disp = fig.add_subplot(gs[0])   # displacement profile
        ax_drift = fig.add_subplot(gs[1])   # storey drift profile
        ax_accel = fig.add_subplot(gs[2])   # acceleration profile
        ax_gm = fig.add_subplot(gs[3])   # ground motion

        # ── Static background elements ───────────────────────────────────────
        # Undeformed centreline
        ax_disp.plot(np.zeros_like(node_z_coords), node_z_coords,
                     color='gray', lw=1.0, alpha=0.6)
        ax_accel.plot(np.zeros_like(node_z_coords), node_z_coords,
                      color='gray', lw=1.0, alpha=0.6)
        # Drift zero-line (staircase x=0)
        x_zero, y_zero = [], []
        for s in range(n_storeys):
            y_zero.extend([node_z_coords[s], node_z_coords[s + 1]])
            x_zero.extend([0.0, 0.0])
        y_zero.append(node_z_coords[-1])
        x_zero.append(0.0)
        ax_drift.plot(x_zero, y_zero, color='gray', lw=1.0, alpha=0.6)

        # Ground motion ghost
        ax_gm.plot(dts, acc, color='lightgray', lw=1.0, alpha=0.8)

        # MinMax collapse marker — draw once as a static artist
        if collapse_time is not None:
            collapse_acc = float(np.interp(collapse_time, dts, acc))
            ax_gm.plot(collapse_time, collapse_acc,
                       marker='x', markersize=12, color='#E53935',
                       markeredgewidth=2.5, zorder=10,
                       label=f'MinMax exceeded ({collapse_time:.2f}s)')
            ax_gm.axvline(collapse_time, color='#E53935', lw=0.8,
                          ls='--', alpha=0.7)
            ax_gm.legend(fontsize=7, loc='upper right')

        # ── Damage state colours ─────────────────────────────────────────────
        # Index 0 = no damage (blue) … index 4 = collapse (red).
        # Per-storey: each storey independently tracks its worst-ever state so
        # colours never regress (a storey that turned red stays red).
        damage_colors = ['#1E88E5', '#43A047', '#FDD835', '#FB8C00', '#E53935']
        n_ds = len(damage_colors)
        # Worst damage state reached so far, per storey (for drift staircase
        # and disp/accel segments) — shape (n_storeys,)
        storey_damage_state = np.zeros(n_storeys, dtype=int)
        # Ground-motion trace uses the worst storey state (structure-level)
        max_damage_state = 0

        # Seed peak annotations from true full-resolution values
        max_drift_val = float(
            np.max(true_peak_drift) *
            100.0) if true_peak_drift is not None else 0.0
        max_accel_val = float(
            np.max(true_peak_accel)) if true_peak_accel is not None else 0.0

        # ── Animated lines — one segment per storey/interval ─────────────────
        # drift staircase: one vertical Line2D per storey + one horizontal
        # connector per inter-storey junction (coloured like the storey below).
        # Base connector: horizontal from 0 → drift[0] at z[0] (ground level).
        drift_lines = [ax_drift.plot([], [], color=damage_colors[0], lw=2.5)[0]
                       for _ in range(n_storeys)]          # verticals
        drift_h_lines = [ax_drift.plot([], [], color=damage_colors[0], lw=2.5)[0]
                         for _ in range(n_storeys)]          # horizontals at top of each storey
        drift_base_line, = ax_drift.plot(
            [], [], color=damage_colors[0], lw=2.5)  # base horizontal
        # displacement profile: one segment per storey interval + node markers
        disp_lines = [ax_disp.plot([], [], 'o-', color=damage_colors[0],
                                   lw=2.0, ms=5)[0]
                      for _ in range(n_storeys)]
        # acceleration profile: one segment per storey interval + node markers
        accel_lines = [ax_accel.plot([], [], 'o-', color=damage_colors[0],
                                     lw=2.0, ms=5)[0]
                       for _ in range(n_storeys)]
        # ground motion trace — single line, coloured by worst global state
        line_gm_trace, = ax_gm.plot([], [], color=damage_colors[0], lw=1.6)

        # ── Axis formatting ──────────────────────────────────────────────────
        ax_disp.set_title('Floor Displacements', fontsize=9, fontweight='bold')
        ax_disp.set_xlabel('Displacement [m]', fontsize=8)
        ax_disp.set_ylabel('Elevation [m]', fontsize=8)
        ax_disp.set_xlim(-max(max_abs_disp * 1.2, 0.01),
                         max(max_abs_disp * 1.2, 0.01))
        ax_disp.set_ylim(ylim_elev)
        ax_disp.grid(True, ls=':', alpha=0.4)
        ax_disp.tick_params(labelsize=8)

        ax_drift.set_title(
            'Storey Drift Profile',
            fontsize=9,
            fontweight='bold')
        ax_drift.set_xlabel(r'Storey Drift [%]', fontsize=8)
        ax_drift.set_ylabel('Elevation [m]', fontsize=8)
        ax_drift.set_xlim(0, xlim_drift)
        ax_drift.set_ylim(ylim_elev)
        ax_drift.grid(True, ls=':', alpha=0.4)
        ax_drift.tick_params(labelsize=8)

        ax_accel.set_title(
            'Floor Accelerations',
            fontsize=9,
            fontweight='bold')
        ax_accel.set_xlabel('Acceleration [g]', fontsize=8)
        ax_accel.set_ylabel('Elevation [m]', fontsize=8)
        ax_accel.set_xlim(-max(max_abs_accel * 1.2, 0.5),
                          max(max_abs_accel * 1.2, 0.5))
        ax_accel.set_ylim(ylim_elev)
        ax_accel.grid(True, ls=':', alpha=0.4)
        ax_accel.tick_params(labelsize=8)

        ax_gm.set_title('Input Ground Motion', fontsize=9, fontweight='bold')
        ax_gm.set_xlabel('Time [s]', fontsize=8)
        ax_gm.set_ylabel('Accel [g]', fontsize=8)
        ax_gm.set_xlim(0, dts[-1])
        ax_gm.set_ylim(np.floor(acc.min()), np.ceil(acc.max()))
        ax_gm.grid(True, ls=':', alpha=0.4)
        ax_gm.tick_params(labelsize=8)

        # Annotations — anchored inside axes with clip_on so they never
        # overflow
        drift_annot = ax_drift.text(
            0.97,
            0.97,
            '',
            transform=ax_drift.transAxes,
            fontsize=7,
            ha='right',
            va='top',
            clip_on=True,
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor='gray',
                alpha=0.7))
        accel_annot = ax_accel.text(
            0.97,
            0.97,
            '',
            transform=ax_accel.transAxes,
            fontsize=7,
            ha='right',
            va='top',
            clip_on=True,
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor='gray',
                alpha=0.7))

        # Threshold lines on drift panel
        if drift_thresholds:
            thr_colors = damage_colors[1:]
            for ti, thr in enumerate(drift_thresholds):
                ax_drift.axvline(thr * 100.0,
                                 color=thr_colors[min(ti, len(thr_colors) - 1)],
                                 lw=0.8, ls='--', alpha=0.7)

        def update(anim_frame):
            nonlocal max_damage_state, max_drift_val, max_accel_val
            frame = int(frame_indices[anim_frame])

            disp_values = nrha_disps[frame, :]
            accel_values = nrha_accels[frame, :]   # already in g

            # ── Per-storey drift [%] ─────────────────────────────────────────
            storey_drifts_pct = (
                np.abs(np.diff(disp_values)) / storey_heights * 100.0
            )
            # Cap on final collapsed frame
            if collapse_time is not None and anim_frame == num_frames - 1:
                storey_drifts_pct = np.clip(storey_drifts_pct, 0.0, xlim_drift)

            # ── Per-storey damage state (cumulative — never regresses) ───────
            if drift_thresholds is not None and len(drift_thresholds) > 0:
                thr_arr = np.array(drift_thresholds)   # ratios, not %
                for s in range(n_storeys):
                    ds = int(np.sum(storey_drifts_pct[s] / 100.0 > thr_arr))
                    if ds > storey_damage_state[s]:
                        storey_damage_state[s] = ds
            # Worst global state drives GM trace colour
            global_state = int(storey_damage_state.max())
            if global_state > max_damage_state:
                max_damage_state = global_state

            # ── Drift staircase ──────────────────────────────────────────────
            # Vertical bars (one per storey) + horizontal connectors at each
            # junction. The horizontal at the top of storey s uses the colour
            # of storey s (the storey below the junction), as requested.
            for s in range(n_storeys):
                c = damage_colors[min(storey_damage_state[s], n_ds - 1)]
                # Vertical: constant x = drift[s], from z[s] to z[s+1]
                drift_lines[s].set_data(
                    [storey_drifts_pct[s], storey_drifts_pct[s]],
                    [node_z_coords[s], node_z_coords[s + 1]]
                )
                drift_lines[s].set_color(c)
                # Horizontal at top of storey s: from drift[s] to drift[s+1]
                # (or to 0 for the topmost storey), at elevation z[s+1].
                next_drift = storey_drifts_pct[s +
                                               1] if s + 1 < n_storeys else 0.0
                drift_h_lines[s].set_data(
                    [storey_drifts_pct[s], next_drift],
                    [node_z_coords[s + 1], node_z_coords[s + 1]]
                )
                drift_h_lines[s].set_color(c)
            # Base horizontal: from 0 to drift[0] at ground elevation z[0]
            c_base = damage_colors[min(storey_damage_state[0], n_ds - 1)]
            drift_base_line.set_data([0.0, storey_drifts_pct[0]],
                                     [node_z_coords[0], node_z_coords[0]])
            drift_base_line.set_color(c_base)

            # ── Displacement profile — one segment per storey interval ───────
            for s in range(n_storeys):
                c = damage_colors[min(storey_damage_state[s], n_ds - 1)]
                disp_lines[s].set_data(
                    [disp_values[s], disp_values[s + 1]],
                    [node_z_coords[s], node_z_coords[s + 1]]
                )
                disp_lines[s].set_color(c)

            # ── Acceleration profile — one segment per storey interval ───────
            for s in range(n_storeys):
                c = damage_colors[min(storey_damage_state[s], n_ds - 1)]
                accel_lines[s].set_data(
                    [accel_values[s], accel_values[s + 1]],
                    [node_z_coords[s], node_z_coords[s + 1]]
                )
                accel_lines[s].set_color(c)

            # ── Ground motion trace ──────────────────────────────────────────
            line_gm_trace.set_data(dts[:frame + 1], acc[:frame + 1])
            line_gm_trace.set_color(
                damage_colors[min(max_damage_state, n_ds - 1)])

            # ── Peak annotations ─────────────────────────────────────────────
            current_drift_max = float(np.max(storey_drifts_pct))
            current_accel_max = float(np.max(np.abs(accel_values)))
            max_drift_val = max(max_drift_val, current_drift_max)
            max_accel_val = max(max_accel_val, current_accel_max)
            drift_annot.set_text(f'Peak drift: {max_drift_val:.2f}%')
            accel_annot.set_text(f'Peak accel: {max_accel_val:.3f} g')

            return (*drift_lines, *drift_h_lines, drift_base_line,
                    *disp_lines, *accel_lines,
                    line_gm_trace, drift_annot, accel_annot)

        ani = FuncAnimation(fig, update, frames=num_frames,
                            interval=10, blit=False, repeat=False)
        if export_path:
            print(f'\nSaving NRHA animation ({num_frames} frames) to: {export_path}')
            try:
                if export_path.lower().endswith('.gif'):
                    ani.save(export_path, writer='pillow', dpi=dpi)
                elif export_path.lower().endswith('.mp4'):
                    try:
                        ani.save(export_path, writer='ffmpeg', dpi=dpi)
                    except Exception:
                        print("FFmpeg not found - falling back to Pillow GIF.")
                        ani.save(export_path.replace('.mp4', '.gif'),
                                 writer='pillow', dpi=dpi)
                else:
                    print("Unknown extension - saving as GIF.")
                    ani.save(export_path + '.gif', writer='pillow', dpi=dpi)
                plt.close(fig)
            except Exception as e:
                print(f'Animation save failed: {e}')
                self._show()
        else:
            self._show()

        return ani

    def plot_demand_profiles(self,
                             peak_drift_list,
                             peak_accel_list,
                             control_nodes,
                             title=None,
                             pFlag=True,
                             export_path=None):
        """
        Generate demand profile plots for peak storey drifts and peak floor accelerations.

        This method creates two side-by-side plots:
        - A plot of peak storey drift (%), displaying how the drift ratio varies with floor number.
        - A plot of peak floor acceleration (g), displaying how the acceleration varies with
          floor number.

        The data is presented as lines representing each control node's response at different
        floors.

        The figure uses ``self.figsize`` with ``constrained_layout`` and is
        saved without ``bbox_inches='tight'`` so that every output image has
        identical, deterministic pixel dimensions.

        Parameters:
        ----------
        peak_drift_list : list of np.ndarray
            A list of arrays where each array contains peak drift values for each floor, with the
            first column being the drift values and the second column being the floor numbers.

        peak_accel_list : list of np.ndarray
            A list of arrays where each array contains peak acceleration values for each floor,
            with the first column being the acceleration values and the second column being the
            floor numbers.

        control_nodes : list
            A list of floor numbers or nodes that represent the control points in the structure.

        title : str, optional
            Custom plot title.

        pFlag : bool, optional, default=True
            If True, the plot is processed (saved/shown).

        export_path : str, optional
            Full path including filename to save the plot. Creates directories if missing.

        Returns:
        --------
        None
            This function saves the plot to a file in the specified output directory.

        """

        # Initialise Plot with two subplots
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=self.figsize, constrained_layout=True)

        # Apply standard styles to subplots
        self._set_plot_style(
            ax1,
            xlabel=r'Peak Storey Drift, $\theta_{max}$ [%]',
            ylabel='Floor No.')
        self._set_plot_style(
            ax2,
            xlabel=r'Peak Floor Acceleration, $a_{max}$ [g]',
            ylabel='Floor No.')

        nst = len(control_nodes) - 1
        for i in range(len(peak_drift_list)):
            # Process and plot Drifts
            x_drift, y_drift = self.duplicate_for_drift(
                peak_drift_list[i][:, 0], control_nodes)
            ax1.plot([float(val) * 100 for val in x_drift], y_drift,
                     linewidth=self.line_widths['medium'],
                     linestyle='solid', color=self.colors['gem'][1], alpha=0.7)

            # Process and plot Accelerations (converted to g)
            ax2.plot([float(val) / 9.81 for val in peak_accel_list[i][:,
                                                                      0]],
                     control_nodes,
                     linewidth=self.line_widths['medium'],
                     linestyle='solid',
                     color=self.colors['gem'][0],
                     alpha=0.7)

        # Axis Customization
        for ax in [ax1, ax2]:
            ax.set_yticks(np.linspace(0, nst, nst + 1))
            ax.set_yticklabels([int(i) for i in np.linspace(0, nst, nst + 1)],
                               fontsize=self.font_sizes['ticks'], fontname=self.font_name)

            # Use smaller font for X-ticks as profiles can be dense
            ax.tick_params(axis='x', labelsize=self.font_sizes['ticks'] - 2)
            ax.set_xlim([0, 5.0])

        # Add title
        default_title = "Seismic Demand Profiles"
        fig.suptitle(title if title else default_title,
                     fontsize=self.font_sizes['title'],
                     fontname=self.font_name)

        # Save or Show
        if pFlag:
            if export_path:
                directory = os.path.dirname(export_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                plt.savefig(export_path, dpi=self.resolution)
                self._show()
            else:
                self._show()
        else:
            plt.close()

    # PLOT MODIFIED CLOUD ANALYSES OUTPUTS

    def plot_mca_analysis(self,
                          cloud_dict,
                          imt_label,
                          edp_label,
                          title=None,
                          pFlag=True,
                          export_path=None):
        """
        Visualizes the Modified Cloud Analysis (MCA) regression including bootstrapping.
        This plot accounts for collapse cases using logistic regression, showing the
        'softening' effect on the median and percentile structural response.

        The figure uses ``self.figsize`` with ``constrained_layout`` and is
        saved without ``bbox_inches='tight'`` so that every output image has
        identical, deterministic pixel dimensions.

        Parameters
        ----------
        cloud_dict : dict
            Output of ``process_mca_results``. Plots cloud data, damage thresholds,
            a fitted regression line, and upper/lower censoring limits in log-log space.

        imt_label : str
            Intensity Measure label for the Y-axis (e.g., ``'PGA [g]'``).

        edp_label : str
            Engineering Demand Parameter label for the X-axis (e.g., ``'PSD [-]'``).

        title : str, optional
            Custom title for the figure. If not provided the title is omitted.

        pFlag : bool, optional
            If ``True`` the plot is displayed or saved. Default is ``True``.

        export_path : str, optional
            Full path including filename to save the plot. Directories are created
            if missing.

        Returns
        -------
        None

        """

        # Setup Data
        inputs = cloud_dict['cloud inputs']
        reg = cloud_dict['regression']
        boot = cloud_dict['bootstraps']
        raw = cloud_dict['raw_data']
        c_limit = inputs['upper_limit']

        # Define Dynamic Range
        all_ims = inputs['imls']
        x_min, x_max = all_ims.min() * 0.8, all_ims.max() * 1.2
        im_vector = np.geomspace(x_min, x_max, 100)

        # Helper Functions for MCA logic
        # Predicted EDP from Cloud: a * IM^b
        def f_edp_nc(a, im, b): return a * (im**b)
        # Predicted Prob. of Collapse from Logistic: 1 / (1 + exp(-(a0 +
        # a1*lnIM)))

        def f_p_coll(a0, a1, im): return 1 / \
            (1 + np.exp(-(a0 + a1 * np.log(im))))
        # MCA median: EDP_nc * exp(sigma * norm_inv(0.5 / (1 - P_collapse)))

        def f_mca(edp, sig, p_c, percentile): return edp * \
            np.exp(sig * norm.ppf(percentile / (1 - p_c)))

        # Initialise Plot
        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)

        # Apply consistent Class Styling
        self._set_plot_style(ax,
                             title=title,
                             xlabel=imt_label,
                             ylabel=edp_label)

        # Plot Bootstrap Samples (Background Cloud)
        n_boot = len(boot['a'])
        for i in range(n_boot):
            p_c_b = f_p_coll(boot['alpha0'][i], boot['alpha1'][i], im_vector)
            # Filter p_c_b to avoid NaNs in norm.ppf (must be < 0.5 for median
            # calculation)
            mask = p_c_b < 0.49
            edp_b = f_edp_nc(boot['a'][i], im_vector[mask], boot['b1'][i])
            mca_b = f_mca(edp_b, boot['sigma_rr'][i], p_c_b[mask], 0.50)

            ax.plot(
                im_vector[mask],
                mca_b,
                color='silver',
                alpha=0.1,
                lw=0.5,
                zorder=1)

        # Plot Mean MCA Regression and Confidence Intervals
        # We use the averaged coefficients stored in 'regression' and
        # 'bootstraps'
        a_m, b_m = np.exp(reg['b0']), reg['b1']  # b0 was stored as log(a)
        sig_m = reg['sigma']
        a0_m, a1_m = boot['alpha0'].mean(), boot['alpha1'].mean()

        p_c_m = f_p_coll(a0_m, a1_m, im_vector)
        # Calculate for percentiles where P(Collapse) hasn't taken over
        mask_m = p_c_m < 0.45
        edp_m = f_edp_nc(a_m, im_vector[mask_m], b_m)

        mca_median = f_mca(edp_m, sig_m, p_c_m[mask_m], 0.50)
        mca_16 = f_mca(edp_m, sig_m, p_c_m[mask_m], 0.16)
        mca_84 = f_mca(edp_m, sig_m, p_c_m[mask_m], 0.84)

        ax.plot(
            im_vector[mask_m],
            mca_median,
            color=self.colors['gem'][1],
            lw=self.line_widths['thick'],
            label='Robust MCA Median',
            zorder=4)
        ax.plot(im_vector[mask_m], mca_16, color=self.colors['gem'][1],
                lw=1, ls='--', label=r'16/84% Percentiles', zorder=4)
        ax.plot(im_vector[mask_m], mca_84, color=self.colors['gem'][1],
                lw=1, ls='--', zorder=4)

        # Plot Raw Data
        ax.scatter(
            raw['im_nc'],
            raw['edp_nc'],
            color=self.colors['gem'][2],
            s=self.marker_sizes['medium'],
            alpha=0.6,
            label='Non-collapse Data',
            zorder=3)
        ax.scatter(raw['im_c'],
                   [c_limit] * len(raw['im_c']),
                   color='darkred',
                   marker='x',
                   s=self.marker_sizes['medium'],
                   label='Collapse Data',
                   zorder=3)

        # Formatting & Limits
        ax.axhline(
            c_limit,
            color='red',
            ls=':',
            lw=1.5,
            label='Collapse Threshold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([inputs['edps'].min() * 0.5, c_limit * 1.5])

        # Text Stats Box
        # a is exp(b0), b is b1, beta is sigma
        a_mean_val = np.exp(reg['b0'])
        b_mean_val = reg['b1']
        beta_val = reg['sigma']

        # alpha means from bootstrap arrays
        a0_m = boot['alpha0'].mean()
        a1_m = boot['alpha1'].mean()

        stats_text = (
            f"Classical Cloud Regression Params:\n"
            f"a-coefficient: {a_mean_val:.2E}\n"
            f"b-coefficient: {b_mean_val:.2f}\n"
            f"beta: {beta_val:.2f}\n"
            f"Logistic Regression Params:\n"
            f"$\\alpha_0$: {a0_m:.2f}\n"
            f"$\\alpha_1$: {a1_m:.2f}"
        )

        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='none'))

        ax.legend(loc='lower right', fontsize=self.font_sizes['legend'])

        # Save or Show
        if pFlag:
            if export_path:
                directory = os.path.dirname(export_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                plt.savefig(export_path, dpi=self.resolution)
                self._show()
                plt.close(fig)
            else:
                self._show()
        else:
            plt.close(fig)

    # PLOT INCREMENTAL DYNAMIC ANALYSES OUTPUTS

    def plot_ida_analysis(self,
                          ida_dict,
                          imt_label,
                          edp_label,
                          xlims,
                          ylims,
                          title=None,
                          pFlag=True,
                          export_path=None):
        """
        Visualizes the Incremental Dynamic Analysis (IDA) suite and statistical summary.

        This method generates a comprehensive IDA plot featuring individual ground motion
        record curves as a background "cloud" and overlays the statistical response
        percentiles. It is designed to provide an immediate visual assessment of
        structural performance across a range of intensities.

        The figure uses ``self.figsize`` with ``constrained_layout`` and is
        saved without ``bbox_inches='tight'`` so that every output image has
        identical, deterministic pixel dimensions.

        Parameters
        ----------
        ida_dict : dict
            The processed results dictionary returned by `postprocessor.process_ida_results`.
            Must contain the following nested keys:
            - ['ida_inputs']['raw_curves']: List of (IM, EDP) pairs for each record.
            - ['ida_inputs']['damage_thresholds']: EDP values for limit states.
            - ['stats']: Dictionary containing 'fitted_edps', 'median_im', 'p16_im', and 'p84_im'.
            - ['ida_inputs']['imt_key']: The label of the intensity measure used.

        imt_label : str
            Intensity Measure Label for the Y-axis (e.g., 'PGA [g]').

        edp_label : str
            Engineering Demand Parameter Label for the X-axis (e.g., 'PSD [-]').

        xlims : tuple of float
            (min, max) limits for the X-axis (EDP axis).

        ylims : tuple of float
            (min, max) limits for the Y-axis (IML axis).

        title : str, optional, default=None
            A custom title for the figure. If not provided, a default title
            incorporating the Intensity Measure (IM) label is used.

        pFlag : bool, optional, default=True
            If True, the plot is processed (saved/shown).

        export_path : str, optional
            Full path including filename to save the plot. Creates directories if missing.


        Returns
        -------
        None
            Displays the matplotlib figure.
        """

        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)

        inputs = ida_dict['ida_inputs']
        stats = ida_dict['stats']

        # 1. Plot Individual Cubic Spline IDA Curves
        for i, curve in enumerate(inputs['raw_curves']):
            ims = curve['im']
            edps = curve['edp']

            if len(ims) > 3:
                # Traditional Vamvatsikos approach:
                # Interpolate a dense IM range to get smooth EDP response
                im_smooth = np.linspace(np.min(ims), np.max(ims), 300)

                # CubicSpline handles the non-monotonic EDP values
                # (the curve can "bend" back and forth on the X-axis)
                cs = CubicSpline(ims, edps)
                edp_smooth = cs(im_smooth)

                ax.plot(edp_smooth, im_smooth, color='gray', alpha=0.15,
                        lw=self.line_widths['thin'], zorder=1,
                        label='Individual Record' if i == 0 else "")
            else:
                ax.plot(edps, ims, '-o', color='gray', alpha=0.15,
                        lw=self.line_widths['thin'], markersize=3, zorder=1)

        # 2. Plot Statistical Percentile Lines
        # Note: Percentiles are usually calculated on the monotonic version
        # to represent "First Collapse" for safety engineering.
        fitted_edps = stats['fitted_edps']

        def plot_stat_line(x, y, color, ls, lw, label):
            mask = ~np.isnan(y)
            if np.sum(mask) > 3:
                # For summary stats, IM is usually the dependent variable
                cs_stat = CubicSpline(x[mask], y[mask])
                x_fine = np.linspace(np.min(x[mask]), np.max(x[mask]), 300)
                ax.plot(
                    x_fine,
                    cs_stat(x_fine),
                    color=color,
                    ls=ls,
                    lw=lw,
                    label=label,
                    zorder=3)

        plot_stat_line(
            fitted_edps,
            stats['p16_im'],
            'green',
            '--',
            self.line_widths['medium'],
            '$16^{th}$ Percentile')
        plot_stat_line(
            fitted_edps,
            stats['median_im'],
            'blue',
            '-',
            self.line_widths['thick'],
            '$50^{th}$ Percentile (Median)')
        plot_stat_line(
            fitted_edps,
            stats['p84_im'],
            'red',
            '--',
            self.line_widths['medium'],
            '$84^{th}$ Percentile')

        # 3. Damage Thresholds and Styling
        ds_colors = self.colors['fragility']
        for i, thresh in enumerate(inputs['damage_thresholds']):
            ax.axvline(thresh,
                       color=ds_colors[i % len(ds_colors)],
                       ls=':',
                       alpha=0.8,
                       lw=self.line_widths['medium'],
                       label=f'$DS_{{{i + 1}}}$ Threshold',
                       zorder=2)

        self._set_plot_style(
            ax, title=title,
            xlabel=edp_label, ylabel=imt_label)

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.legend(loc='upper right', fontsize=self.font_sizes['legend'])

        # Save or Show
        if pFlag:
            if export_path:
                directory = os.path.dirname(export_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                plt.savefig(export_path, dpi=self.resolution)
                self._show()
                plt.close(fig)
            else:
                self._show()
        else:
            plt.close(fig)

    # PLOT MULTIPLE STRIPE ANALYSES OUTPUTS

    def plot_msa_analysis(self,
                          stripe_imls,
                          stripe_edps,
                          imt_label,
                          edp_label,
                          xlims,
                          ylims,
                          title=None,
                          pFlag=True,
                          export_path=None):
        """
        Visualizes Multiple Stripe Analysis (MSA) results.

        For each intensity stripe the method plots:
        - Individual ground-motion response points coloured and sized by IM level.
        - A filled lognormal PDF silhouette scaled to the inter-stripe spacing.
        - A vertical line at the lognormal median and dashed lines at the 16th/84th percentiles, both labelled on the first stripe only.
        - A horizontal bracket connecting the 16th and 84th percentile ticks per stripe.

        Parameters
        ----------
        stripe_imls : 2D array
            Matrix of IM levels (n_gmrs × n_stripes); only the first row is used
            as the unique IM level for each stripe.
        stripe_edps : 2D array
            Matrix of EDP responses (n_gmrs × n_stripes) as dimensionless ratios.
            Converted to percent internally.
        imt_label : str
            Y-axis label (e.g. 'Sa(T1) [g]').
        edp_label : str
            X-axis label (e.g. 'Peak Inter-Storey Drift').  '[%]' is appended.
        xlims : tuple of float
            (min, max) for the EDP (x) axis.
        ylims : tuple of float
            (min, max) for the IM (y) axis.
        title : str, optional
            Figure title.  Defaults to a standard MSA title.
        pFlag : bool, default True
            Show/save the figure when True; close silently when False.
        export_path : str, optional
            Full file path to save the figure.
        """
        # ── Data preparation ─────────────────────────────────────────────────
        stripe_edps = np.asarray(stripe_edps, dtype=float) * 100.0   # → %
        stripe_imls = np.asarray(stripe_imls, dtype=float)
        num_gmrs, num_stripes = stripe_edps.shape
        # 1-D IM levels
        unique_imls = stripe_imls[0, :]

        # Colour map: low IM = cool blue, high IM = warm red
        cmap = plt.cm.RdYlBu_r
        norm = mcolors.Normalize(
            vmin=unique_imls.min(),
            vmax=unique_imls.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Inter-stripe spacing for PDF height scaling
        if num_stripes > 1:
            stripe_gap = np.diff(unique_imls).min() * 0.72
        else:
            stripe_gap = unique_imls[0] * 0.20

        # ── Figure / axes ────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)

        # ── Per-stripe rendering ─────────────────────────────────────────────
        for j in range(num_stripes):
            im_level = unique_imls[j]
            edp_values = stripe_edps[:, j]
            stripe_col = cmap(norm(im_level))

            # --- scatter: individual GM points ---
            ax.scatter(edp_values, np.full(num_gmrs, im_level),
                       color=stripe_col, s=50, alpha=0.70, zorder=3,
                       linewidths=0.5, edgecolors='white',
                       label='Ground-motion results' if j == 0 else '')

            # --- lognormal fit ---
            valid = edp_values[edp_values > 0]
            if len(valid) < 2:
                continue

            log_v = np.log(valid)
            mu_ln = log_v.mean()
            sigma_ln = max(log_v.std(ddof=1), 1e-6)
            median_v = np.exp(mu_ln)
            p16_v = np.exp(mu_ln - sigma_ln)
            p84_v = np.exp(mu_ln + sigma_ln)

            x_lo = max(xlims[0] * 0.5, valid.min() * 0.3, 0.001)
            x_hi = min(xlims[1] * 1.5, valid.max() * 2.0)
            x_pdf = np.linspace(x_lo, x_hi, 500)
            pdf_v = stats.lognorm.pdf(x_pdf, s=sigma_ln, scale=np.exp(mu_ln))
            pdf_s = (pdf_v / pdf_v.max()) * stripe_gap * 0.85

            # filled silhouette
            ax.fill_betweenx(im_level + pdf_s, x_pdf,
                             np.full_like(x_pdf, im_level),
                             where=(im_level + pdf_s > im_level),
                             facecolor=stripe_col, alpha=0.25, zorder=2)
            # crisp outline
            ax.plot(x_pdf, im_level + pdf_s, color=stripe_col,
                    lw=2.0, alpha=0.95, zorder=4)

            # median line — full PDF height
            ax.vlines(median_v, im_level, im_level + stripe_gap * 0.85,
                      color=stripe_col, lw=2.5, zorder=5,
                      label='Lognormal median' if j == 0 else '')

            # 16th / 84th percentile ticks — 55% of PDF height
            for pval in (p16_v, p84_v):
                ax.vlines(pval, im_level, im_level + stripe_gap * 0.50,
                          color=stripe_col, lw=1.5, ls='--', zorder=5,
                          label=(r'16$^{\rm th}$ / 84$^{\rm th}$ percentile'
                                 if j == 0 and pval == p16_v else ''))

            # horizontal bracket connecting 16th-84th at 50% height
            ax.hlines(im_level + stripe_gap * 0.50, p16_v, p84_v,
                      color=stripe_col, lw=1.2, ls='--', alpha=0.75, zorder=4)

        # ── Styling ──────────────────────────────────────────────────────────
        self._set_plot_style(ax,
                             title=title,
                             xlabel=f"{edp_label} [%]",
                             ylabel=imt_label)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        # ── Legend ───────────────────────────────────────────────────────────
        ax.legend(loc='upper right', fontsize=self.font_sizes['legend'])

        # ── Save / show ──────────────────────────────────────────────────────
        if pFlag:
            if export_path:
                directory = os.path.dirname(export_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                plt.savefig(export_path, dpi=self.resolution)
                self._show()
                plt.close(fig)
            else:
                self._show()
        else:
            plt.close(fig)

    # PLOT MODIFIED CLOUD ANALYSES FRAGILITY OUTPUTS

    def plot_fragility_from_mca(self,
                                cloud_dict,
                                imt_label,
                                xlims,
                                ylims,
                                title=None,
                                cloud_method=None,
                                plot_bootstrap=False,
                                pFlag=True,
                                export_path=None):
        """
        Generates a fragility analysis plot showing the Probability of
        Exceedance (PoE) for multiple damage states using Modified Cloud
        Analysis (MCA) results.

        Supports both uncertainty quantification approaches:

        - ``'bootstrap'``: plots the mean fragility curves and,
          optionally, individual bootstrap realizations as a faint
          background cloud.
        - ``'classical'``: plots the Robust Fragility mean and a
          shaded ±k·σ confidence band derived from the posterior
          variance (Jalayer et al. 2017).

        The ``cloud_method`` is auto-detected from
        ``cloud_dict['fragility']['cloud_method']`` when the argument is
        ``None`` (default), falling back to ``'bootstrap'`` for dicts
        produced by older code that does not carry the key.

        The figure uses ``self.figsize`` with ``constrained_layout`` and
        is saved without ``bbox_inches='tight'`` so that every output
        image has identical, deterministic pixel dimensions.

        Parameters
        ----------
        cloud_dict : dict
            Standardized dictionary returned by
            ``postprocessor.process_mca_results()``. Required keys:

            - ``'fragility'``: contains ``intensities`` (1-D array),
              ``poes`` (2-D), ``medians``, ``betas_total``.
            - For ``'bootstrap'``: ``'bootstraps'`` sub-dict with
              ``alpha0``, ``alpha1`` arrays and optional ``poes_all``
              (3-D array of per-iteration curves).
            - For ``'classical'``: ``'fragility'`` must also contain
              ``poes_robust_plus`` and ``poes_robust_minus``; and
              ``'mcmc'`` sub-dict with ``chi_mean``.

        imt_label : str
            X-axis label (e.g., ``'PGA [g]'``, ``'Sa(T1) [g]'``).

        xlims : tuple of float
            ``(min, max)`` limits for the X-axis.

        ylims : tuple of float
            ``(min, max)`` limits for the Y-axis.

        title : str, optional
            Custom plot title. If ``None``, a method-specific default
            is used.

        cloud_method : {'bootstrap', 'classical'} or None, optional
            Selects the plotting style. When ``None`` (default), the
            value is read from
            ``cloud_dict['fragility'].get('cloud_method', 'bootstrap')``.

        plot_bootstrap : bool, default False
            *Bootstrap only.* If ``True``, plots all individual
            bootstrap fragility curves with low alpha as a background
            cloud. Ignored when ``cloud_method='classical'``.

        pFlag : bool, default True
            If ``True``, renders and/or saves the figure. If ``False``,
            closes the figure immediately (useful for batch runs).

        export_path : str, optional
            Full file path (including extension) for saving the figure.
            The parent directory is created automatically if needed.

        Returns
        -------
        None
        """

        # -----------------------------------------------------------------
        # Resolve cloud_method
        # -----------------------------------------------------------------
        if cloud_method is None:
            cloud_method = cloud_dict['fragility'].get(
                'cloud_method', 'bootstrap')

        frag = cloud_dict['fragility']
        intensities = frag['intensities']
        poes_mean = frag['poes']
        medians = frag['medians']
        betas = frag['betas_total']
        n_ds = poes_mean.shape[1]
        n_fragility_colors = len(self.colors['fragility'])

        # -----------------------------------------------------------------
        # Initialise figure
        # -----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)
        self._set_plot_style(
            ax,
            xlabel=imt_label,
            ylabel=r'Probability of Exceedance $P(DS \geq ds | IM)$')

        # -----------------------------------------------------------------
        # Bootstrap rendering
        # -----------------------------------------------------------------
        if cloud_method.lower() == 'bootstrap':
            boot = cloud_dict['bootstraps']
            boot_poes = boot.get('poes_all', [])
            alpha0_mean = boot['alpha0'].mean()
            alpha1_mean = boot['alpha1'].mean()

            # Optional: individual bootstrap curves as faint background
            if plot_bootstrap and len(boot_poes) > 0:
                for i in range(len(boot_poes)):
                    for ds in range(n_ds):
                        c_bg = (
                            'black' if ds == n_ds - 1
                            else self.colors['fragility'][
                                ds % n_fragility_colors])
                        ax.plot(intensities, boot_poes[i][:, ds],
                                color=c_bg, alpha=0.05,
                                lw=0.5, zorder=1)

            # Mean fragility curves
            for ds in range(n_ds):
                if ds == n_ds - 1:
                    c = 'black'
                    label = (
                        rf"Collapse: $\alpha_0$={alpha0_mean:.2f}, "
                        rf"$\alpha_1$={alpha1_mean:.2f}")
                else:
                    c = self.colors['fragility'][ds % n_fragility_colors]
                    label = (
                        rf"DS{ds + 1}: $\theta$={medians[ds]:.2f}g, "
                        rf"$\beta$={betas[ds]:.2f}")
                ax.plot(intensities, poes_mean[:, ds],
                        color=c,
                        linewidth=self.line_widths['thick'],
                        label=label,
                        zorder=3)

            default_title = (
                "Fragility Functions from Modified Cloud Analysis"
                " (Bootstrap)")

        # -----------------------------------------------------------------
        # Classical (Bayesian MCMC) rendering
        # -----------------------------------------------------------------
        elif cloud_method.lower() == 'classical':
            poes_plus = frag['poes_robust_plus']
            poes_minus = frag['poes_robust_minus']
            k = frag.get('confidence_k', 2)

            mc = cloud_dict.get('mcmc', {})
            chi_mean = mc.get('chi_mean', np.zeros(5))
            alpha0_mean = float(chi_mean[3])
            alpha1_mean = float(chi_mean[4])

            for ds in range(n_ds):
                if ds == n_ds - 1:
                    c = 'black'
                    label = (
                        rf"Collapse: $\alpha_0$={alpha0_mean:.2f}, "
                        rf"$\alpha_1$={alpha1_mean:.2f}")
                else:
                    c = self.colors['fragility'][ds % n_fragility_colors]
                    label = (
                        rf"DS{ds + 1}: $\theta$={medians[ds]:.2f}g, "
                        rf"$\beta$={betas[ds]:.2f}")

                # Confidence band (±k·σ)
                ax.fill_between(
                    intensities,
                    poes_minus[:, ds], poes_plus[:, ds],
                    color=c, alpha=0.15, zorder=1,
                    label=(rf"$\pm{k}\sigma$ band" if ds == 0
                           else None))

                # Robust mean curve
                ax.plot(intensities, poes_mean[:, ds],
                        color=c,
                        linewidth=self.line_widths['thick'],
                        label=label,
                        zorder=3)

            default_title = (
                "Robust Fragility Functions from Modified Cloud Analysis"
                " (Classical / Jalayer et al. 2017)")

        else:
            raise ValueError(
                f"Unknown cloud_method '{cloud_method}'. "
                f"Expected 'bootstrap' or 'classical'.")

        # -----------------------------------------------------------------
        # Final formatting
        # -----------------------------------------------------------------
        ax.set_xlim([xlims[0], xlims[1]])
        ax.set_ylim([ylims[0], ylims[1]])
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.set_title(
            title if title else default_title,
            fontsize=self.font_sizes['title'])
        ax.legend(
            loc='lower right',
            fontsize=self.font_sizes['legend'],
            frameon=True,
            framealpha=0.9,
            edgecolor='black')

        # Save or show
        if pFlag:
            if export_path:
                directory = os.path.dirname(export_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                plt.savefig(export_path, dpi=self.resolution)
                self._show()
                plt.close(fig)
            else:
                self._show()
        else:
            plt.close(fig)

    # PLOT INCREMENTAL DYNAMIC ANALYSES FRAGILITY OUTPUTS

    def plot_fragility_from_ida(self,
                                ida_dict,
                                imt_label,
                                xlims,
                                ylims,
                                title=None,
                                pFlag=True,
                                export_path=None):
        """
        Generate a fragility analysis plot showing the probability of exceedance (PoE)
        for multiple damage states derived from IDA results.
        This method visualizes the lognormal fragility functions fitted during the
        Incremental Dynamic Analysis. Each curve represents the probability that a
        specific engineering demand parameter (EDP) threshold (e.g., drift limit)
        is exceeded given a specific intensity measure (IM) level.

        The figure uses ``self.figsize`` with ``constrained_layout`` and is
        saved without ``bbox_inches='tight'`` so that every output image has
        identical, deterministic pixel dimensions.

        Parameters
        ----------
        ida_dict : dict
            Nested dictionary from ``process_ida_results``. Must contain
            ``'fragility'`` (with keys ``'intensities'``, ``'poes'``, ``'medians'``)
            and ``'ida_inputs'`` (with key ``'imt_key'``).


        imt_label : str
            Label for the X-axis (e.g., 'PGA [g]').

        xlims : tuple of float
            (min, max) limits for the X-axis (EDP axis).

        ylims : tuple of float
            (min, max) limits for the Y-axis (Probability axis).

        title : str, optional
            Custom plot title.

        pFlag : bool, optional, default=True
            If True, the plot is processed (saved/shown).

        export_path : str, optional
            Full path including filename to save the plot. Creates directories if missing.

        Returns
        -------
        None
            The method renders the plot using Matplotlib and handles saving
            via the internal `_save_plot` utility.
        """

        # Setup Data
        frag_data = ida_dict['fragility']
        intensities = frag_data['intensities']
        # 2D array [n_intensities x n_thresholds]
        poes = frag_data['poes']
        medians = frag_data['medians']
        betas = frag_data['betas_total']

        # Initialize Plot
        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)

        default_title = f"Fragility Functions for {imt_label}"
        self._set_plot_style(
            ax,
            title=title if title else default_title,
            xlabel=imt_label,
            ylabel=r'Probability of Exceedance $P(DS \geq ds | IM)$')

        n_ds = poes.shape[1]

        # 3. Plot Fragility Curves and Empirical Points
        for ds in range(n_ds):
            # Assign color
            color = self.colors['fragility'][ds %
                                             len(self.colors['fragility'])]
            label_prefix = f"DS{ds + 1}"

            # Plot Continuous Lognormal Curve
            ax.plot(intensities, poes[:, ds],
                    color=color,
                    linewidth=self.line_widths['thick'],
                    label=rf"{label_prefix}: $\theta$={medians[ds]:.2f}g, $\beta$={betas[ds]:.2f}",
                    zorder=3)

        ax.set_xlim([xlims[0], xlims[1]])
        ax.set_ylim([ylims[0], ylims[1]])
        ax.legend(
            fontsize=self.font_sizes['legend'],
            loc='lower right',
            frameon=True)

        # Save or Show
        if pFlag:
            if export_path:
                # Save to disk
                directory = os.path.dirname(export_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                plt.savefig(export_path, dpi=self.resolution)
                self._show()
            # Show if no path OR if you want to see it after saving
            if not export_path:
                # Display but do not save to disk
                self._show()
            else:
                # Close the plot to free memory after saving if not showing
                plt.close()

    # PLOT MULTIPLE STRIPE ANALYSES FRAGILITY OUTPUTS

    def plot_fragility_from_msa(self,
                                msa_dict,
                                imt_label,
                                xlims,
                                ylims,
                                title=None,
                                pFlag=True,
                                export_path=None):
        """
        Generate a fragility analysis plot showing the probability of exceedance (PoE)
        for multiple damage states derived from MSA results.
        This method visualizes the lognormal fragility functions fitted during the
        Multiple Stripe Analysis. Each curve represents the probability that a
        specific engineering demand parameter (EDP) threshold (e.g., drift limit)
        is exceeded given a specific intensity measure (IM) level.

        The figure uses ``self.figsize`` with ``constrained_layout`` and is
        saved without ``bbox_inches='tight'`` so that every output image has
        identical, deterministic pixel dimensions.

        Parameters
        ----------
        msa_dict : dict
            Dictionary containing:
            - 'fragility': {'intensities': [], 'poes': [], 'medians': [], 'betas': []}
            - 'metadata': {'stripe_levels': [], 'observed_fractions': []}

        imt_label : str
            Label for the X-axis (Intensity Measure).

        xlims : tuple of float
            (min, max) limits for the X-axis (EDP axis).

        ylims : tuple of float
            (min, max) limits for the Y-axis (Probability axis).

        title : str, optional
            Custom title for the plot.

        pFlag : bool, default True
            Render and show/save the plot.

        export_path : str, optional
            Path to export the plot.
        """
        # 1. Setup Data
        frag = msa_dict['fragility']
        meta = msa_dict.get('metadata', {})

        intensities = frag['intensities']
        poes_mean = frag['poes']  # 2D array [IM x DamageState]
        medians = frag['medians']
        betas = frag['betas_total']

        # Empirical data (The actual fractions from the stripes)
        stripe_levels = meta.get('stripe_levels', [])
        observed_fractions = meta.get(
            'observed_fractions',
            [])  # List of arrays per DS

        # 2. Initialise Plot
        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)
        self._set_plot_style(
            ax,
            xlabel=imt_label,
            ylabel=r'Probability of Exceedance $P(DS \geq ds | IM)$')

        n_ds = poes_mean.shape[1]

        # 3. Plot Fragility Curves and Empirical Points
        for ds in range(n_ds):
            # Assign color
            color = self.colors['fragility'][ds %
                                             len(self.colors['fragility'])]
            label_prefix = f"DS{ds + 1}"

            # Plot Continuous Lognormal Curve (MLE Fit)
            ax.plot(intensities, poes_mean[:, ds],
                    color=color,
                    linewidth=self.line_widths['thick'],
                    label=rf"{label_prefix}: $\theta$={medians[ds]:.2f}g, $\beta$={betas[ds]:.2f}",
                    zorder=3)

            # Plot Discrete Empirical Points (Observed at each stripe)
            if len(observed_fractions) > 0:
                ax.scatter(stripe_levels, observed_fractions[ds],
                           color=color, marker='o', s=60, edgecolors='white',
                           linewidth=1, zorder=4, alpha=0.8)

        # 4. Final Formatting
        ax.set_xlim([xlims[0], xlims[1]])
        ax.set_ylim([ylims[0], ylims[1]])
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        default_title = "Fragility Functions from Multiple Stripe Analysis (MLE)"
        ax.set_title(
            title if title else default_title,
            fontsize=self.font_sizes['title'])
        ax.legend(loc='lower right', fontsize=self.font_sizes['legend'],
                  frameon=True, framealpha=0.9, edgecolor='black')

        # Save or Show
        if pFlag:
            if export_path:
                # Save to disk
                directory = os.path.dirname(export_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                plt.savefig(export_path, dpi=self.resolution)
                self._show()
            # Show if no path OR if you want to see it after saving
            if not export_path:
                # Display but do not save to disk
                self._show()
            else:
                # Close the plot to free memory after saving if not showing
                plt.close()

    # PLOT STOREY LOSS FUNCTION GENERATOR OUTPUT

    def plot_slf_model(self,
                       out,
                       cache,
                       edp_label,
                       loss_label,
                       xlims,
                       ylims,
                       title=None,
                       pFlag=True,
                       export_path=None):
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

        The figure uses ``self.figsize`` with ``constrained_layout`` and is
        saved without ``bbox_inches='tight'`` so that every output image has
        identical, deterministic pixel dimensions.

        Parameters
        ----------
        out : dict
            Results dictionary with keys ``'edp_range'`` and ``'slf'``
            (the fitted Storey Loss Function curve).

        cache : dict
            Cached simulation data with keys ``'total_loss_storey'``,
            ``'empirical_16th'``, ``'empirical_84th'``, and ``'empirical_median'``.

        edp_label : str
            Label for the x-axis (Engineering Demand Parameter).

        loss_label : str
            Label for the y-axis (Storey Loss Ratio).

        xlims : tuple of float
            (min, max) limits for the X-axis (EDP axis).

        ylims : tuple of float
            (min, max) limits for the Y-axis (Loss axis).

        title : str, optional
            Custom plot title.

        pFlag : bool, optional
            If ``True`` the plot is displayed or saved. Default is ``True``.

        export_path : str, optional
            Full path including filename to save the plot. Directories are created
            if missing.

        Returns
        -------
        None
        """
        keys_list = list(cache.keys())
        for i, current_key in enumerate(keys_list):
            rlz = len(cache[current_key]['total_loss_storey'])
            total_loss_storey_array = np.array(
                [cache[current_key]['total_loss_storey'][i] for i in range(rlz)])

            fig, ax = plt.subplots(
                figsize=self.figsize, constrained_layout=True)
            self._set_plot_style(ax, xlabel=edp_label, ylabel='Storey Loss')

            for i in range(rlz):
                ax.scatter(out[current_key]['edp_range'],
                           total_loss_storey_array[i,
                                                   :],
                           color=self.colors['gem'][3],
                           s=self.marker_sizes['small'],
                           alpha=0.5)

            ax.fill_between(
                out[current_key]['edp_range'],
                cache[current_key]['empirical_16th'],
                cache[current_key]['empirical_84th'],
                color='gray',
                alpha=0.3,
                label=r'16$^{\text{th}}$-84$^{\text{th}}$ Percentile')
            ax.plot(
                out[current_key]['edp_range'],
                cache[current_key]['empirical_median'],
                lw=self.line_widths['medium'],
                color='blue',
                label='Simulations - Median')
            ax.plot(
                out[current_key]['edp_range'],
                out[current_key]['slf'],
                color='black',
                lw=self.line_widths['medium'],
                label='SLF - Fitted')

            ax.legend(fontsize=self.font_sizes['legend'])

        self._set_plot_style(ax,
                             title=title or "Storey Loss Function",
                             xlabel=edp_label,
                             ylabel=loss_label)

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.legend(loc='upper right', fontsize=self.font_sizes['legend'])

        # Save or Show
        if pFlag:
            if export_path:
                directory = os.path.dirname(export_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                plt.savefig(export_path, dpi=self.resolution)
                self._show()
            else:
                self._show()
        else:
            plt.close()

    # PLOT VULNERABILITY FUNCTIONS

    def plot_vulnerability_function(self,
                                    intensities,
                                    loss,
                                    cov,
                                    imt_label,
                                    loss_label,
                                    title=None,
                                    pFlag=True,
                                    export_path=None):
        """
        Generate a vulnerability analysis plot featuring Beta distributions and a mean loss curve.

        This method visualizes the uncertainty in loss ratios across different seismic intensities.
        It simulates Beta distributions based on mean loss and CoV, rendering them as
        truncated violin plots (strictly bounded 0-1) to represent the physical limits
        of structural damage.

        The figure uses ``self.figsize`` with ``constrained_layout`` and is
        saved without ``bbox_inches='tight'`` so that every output image has
        identical, deterministic pixel dimensions.

        Parameters
        ----------
        intensities : list of float
            Intensity Measure (IM) levels (e.g., PGA, Sa) analyzed.

        loss : list of float
            Mean loss ratios (0.0 to 1.0) corresponding to each intensity.

        cov : list of float
            Coefficient of Variation for loss at each intensity.

        imt_label : str
            Label for the X-axis (e.g., 'PGA [g]').

        loss_label : str
            Label for the primary Y-axis loss curve (e.g., 'Mean Damage Ratio').

        title : str, optional
            Custom plot title.

        pFlag : bool, optional, default=True
            If True, the plot is processed (saved/shown).

        export_path : str, optional
            Full path including filename to save the plot. Creates directories if missing.

        Returns
        -------
        None
        """
        # Simulating Beta distributions for each intensity measure
        simulated_data = []
        intensity_labels = []

        for j, mean_loss in enumerate(loss):
            # Beta distribution requires mean in (0, 1)
            mu = np.clip(mean_loss, 0.0001, 0.9999)
            variance = (cov[j] * mu) ** 2

            # Constraint: Variance must be < mu * (1 - mu)
            limit = mu * (1 - mu)
            if variance >= limit:
                variance = limit * 0.99  # Cap variance to allow distribution fitting

            alpha = mu * (mu * (1 - mu) / variance - 1)
            beta_param = (1 - mu) * (mu * (1 - mu) / variance - 1)

            # Generate samples and clip to ensure physical bounds
            data = np.random.beta(alpha, beta_param, 10000)
            simulated_data.append(data)
            intensity_labels.extend([intensities[j]] * len(data))

        # We name the column 'Loss_Val' for consistent reference in Seaborn
        df_sns = pd.DataFrame({'Intensity Measure': intensity_labels,
                               'Loss_Val': np.concatenate(simulated_data)})

        # Initialise Plot
        fig, ax1 = plt.subplots(figsize=self.figsize, constrained_layout=True)

        # Set plot style
        default_title = f"Vulnerability Function: {imt_label}"
        self._set_plot_style(ax1,
                             title=title if title else default_title,
                             xlabel=imt_label,
                             ylabel=loss_label)

        # Violin plot for Beta distributions
        # We use 'Loss_Val' to match the DataFrame column
        sns.violinplot(x='Intensity Measure', y='Loss_Val', data=df_sns,
                       density_norm='width', bw_method=0.2,
                       cut=0,
                       inner=None,
                       ax=ax1,
                       zorder=1,
                       color='skyblue')

        # Overlay a strip plot for sample density
        sns.stripplot(
            x='Intensity Measure',
            y='Loss_Val',
            data=df_sns,
            color='black',
            size=1,
            alpha=0.2,
            ax=ax1,
            zorder=2)

        # Style the primary axis (Distributions)
        ax1.set_ylim(0, 1.0)
        ax1.yaxis.label.set_color('blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Secondary Axis for the Mean Loss Curve
        ax2 = ax1.twinx()
        ax2.plot(
            range(
                len(intensities)),
            loss,
            marker='s',
            ls='-',
            color='red',
            lw=self.line_widths['medium'],
            label="Mean Loss Ratio",
            zorder=5)

        # Style the secondary axis (Loss Curve)
        ax2.set_ylabel(
            loss_label,
            color='red',
            rotation=270,
            labelpad=20,
            fontsize=self.font_sizes['labels'],
            fontname=self.font_name)
        ax2.tick_params(
            axis='y',
            labelcolor='red',
            labelsize=self.font_sizes['ticks'])
        ax2.set_ylim(0, 1.0)

        # Sync X-axis ticks with intensity values
        ax1.set_xticks(range(len(intensities)))
        ax1.set_xticklabels([f"{x:.3f}" for x in intensities],
                            rotation=45, ha='right', fontsize=8)

        # Combined Legend
        beta_patch = mpatches.Patch(
            color='skyblue',
            label="Beta Distribution (Uncertainty)")
        ax1.legend(
            handles=[beta_patch],
            loc='upper left',
            fontsize=self.font_sizes['legend'])
        ax2.legend(loc='upper left', bbox_to_anchor=(
            0, 0.93), fontsize=self.font_sizes['legend'])

        # Save or show
        if pFlag:
            if export_path:
                directory = os.path.dirname(export_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                plt.savefig(export_path, dpi=self.resolution)
                self._show()
            else:
                self._show()
        else:
            plt.close()
