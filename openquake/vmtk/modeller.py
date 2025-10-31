
import os
import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation

class modeller():
    """
    A class to model and analyze multi-degree-of-freedom (MDOF) oscillators using OpenSees.

    This class provides functionality to create, analyze, and visualize structural models
    for dynamic and static analyses, including gravity analysis, modal analysis, static
    pushover analysis, cyclic pushover analysis, and nonlinear time-history analysis.

    Attributes
    ----------
    number_storeys : int
        The number of storeys in the building model.
    floor_heights : list
        List of floor heights in meters.
    floor_masses : list
        List of floor masses in tonnes.
    storey_disps : np.array
        Array of storey displacements (size = number of storeys, CapPoints).
    storey_forces : np.array
        Array of storey forces (size = number of storeys, CapPoints).
    degradation : bool
        Boolean to enable or disable hysteresis degradation.

    Methods
    -------
    __init__(number_storeys, floor_heights, floor_masses, storey_disps, storey_forces, degradation)
        Initializes the modeller object and validates input parameters.
    create_Pinching4_material(mat1Tag, mat2Tag, storey_forces, storey_disps, degradation)
        Creates a Pinching4 material model for the MDOF oscillator.
    compile_model()
        Compiles and sets up the MDOF oscillator model in OpenSees.
    plot_model(display_info=True)
        Plots the 3D visualization of the OpenSees model.
    do_gravity_analysis(nG=100, ansys_soe='UmfPack', constraints_handler='Transformation', numberer='RCM', test_type='NormDispIncr', init_tol=1.0e-6, init_iter=500, algorithm_type='Newton', integrator='LoadControl', analysis='Static')
        Performs gravity analysis on the MDOF system.
    do_modal_analysis(num_modes=3, solver='-genBandArpack', doRayleigh=False, pflag=False)
        Performs modal analysis to determine natural frequencies and mode shapes.
    do_spo_analysis(ref_disp, disp_scale_factor, push_dir, phi, pflag=True, num_steps=200, ansys_soe='BandGeneral', constraints_handler='Transformation', numberer='RCM', test_type='EnergyIncr', init_tol=1.0e-5, init_iter=1000, algorithm_type='KrylovNewton')
        Performs static pushover analysis (SPO) on the MDOF system.
    do_cpo_analysis(ref_disp, mu_levels, push_dir, dispIncr, pflag=True, num_steps=200, ansys_soe='BandGeneral', constraints_handler='Transformation', numberer='RCM', test_type='NormDispIncr', init_tol=1.0e-5, init_iter=1000, algorithm_type='KrylovNewton')
        Performs cyclic pushover analysis (CPO) on the MDOF system.
    do_nrha_analysis(fnames, dt_gm, sf, t_max, dt_ansys, nrha_outdir, pflag=True, xi=0.05, ansys_soe='BandGeneral', constraints_handler='Plain', numberer='RCM', test_type='NormDispIncr', init_tol=1.0e-6, init_iter=50, algorithm_type='Newton')
        Performs nonlinear time-history analysis (NRHA) on the MDOF system.

    """
    def __init__(self, number_storeys, floor_heights, floor_masses, storey_disps, storey_forces, degradation):
        """
        Initializes the modeller object and validates the input parameters.

        Parameters
        ----------
        number_storeys : int
            The number of storeys in the building model.
        floor_heights : list
            List of floor heights in meters (e.g., [2.5, 3.0]).
        floor_masses : list
            List of floor masses in tonnes (e.g., [1000, 1200]).
        storey_disps : np.array
            Array of storey displacements (size = number of storeys, CapPoints).
        storey_forces : np.array
            Array of storey forces (size = number of storeys, CapPoints).
        degradation : bool
            Boolean to enable or disable hysteresis degradation.

        Raises
        ------
        ValueError
            If the number of entries in `floor_heights` or `floor_masses` does not match `number_storeys`.
        """

        ### Run tests on input parameters
        if len(floor_heights)!=number_storeys or len(floor_masses)!=number_storeys:
            raise ValueError('Number of entries exceed the number of storeys!')

        self.number_storeys = number_storeys
        self.floor_heights  = floor_heights
        self.floor_masses   = floor_masses
        self.storey_disps   = storey_disps
        self.storey_forces  = storey_forces
        self.degradation    = degradation


    def create_Pinching4_material(self, mat1Tag, mat2Tag, storey_forces, storey_disps, degradation):
        """
        Creates a Pinching4 material model for the multi-degree-of-freedom material object in stick model analysis.

        The Pinching4 material model is used to simulate hysteretic behavior in structures under dynamic loading,
        including degradation if enabled. The method assigns the material properties to the building storeys based
        on the given parameters.

        Parameters
        ----------
        mat1Tag : int
            Material tag for the first material in the Pinching4 model.
        mat2Tag : int
            Material tag for the second material in the Pinching4 model.
        storey_forces : np.array
            Array of storey forces at each storey in the model.
        storey_disps : np.array
            Array of storey displacements corresponding to the forces.
        degradation : bool
            Boolean flag to enable or disable hysteresis degradation in the Pinching4 material model.

        Returns
        -------
        None
            This method does not return any value but modifies the internal material definitions for the model.

        References:
        -----------
        1) Vamvatsikos D (2011) Software—earthquake, steel dynamics and probability, viewed January 2021.
        http://users.ntua.gr/divamva/software.html

        2) Martins, L., Silva, V., Crowley, H. et al. Vulnerability modellers toolkit, an open-source platform
        for vulnerability analysis. Bull Earthquake Eng 19, 5691–5709 (2021). https://doi.org/10.1007/s10518-021-01187-w

        3) Minjie Zhu, Frank McKenna, Michael H. Scott, OpenSeesPy: Python library for the OpenSees finite element framework,
        SoftwareX, Volume 7, 2018, Pages 6-11, ISSN 2352-7110, https://doi.org/10.1016/j.softx.2017.10.009.
        (https://www.sciencedirect.com/science/article/pii/S2352711017300584)

        Notes
        -----
        The `mat1Tag` and `mat2Tag` represent different materials used in the Pinching4 hysteretic model,
        where the degradation flag controls the material's degradation behavior during the simulation.
        """

        force=np.zeros([5,1])
        disp =np.zeros([5,1])

        # Bilinear
        if len(storey_forces)==2:
              #bilinear curve
              force[1]=storey_forces[0]
              force[4]=storey_forces[-1]

              disp[1]=storey_disps[0]
              disp[4]=storey_disps[-1]

              disp[2]=disp[1]+(disp[4]-disp[1])/3
              disp[3]=disp[1]+2*((disp[4]-disp[1])/3)

              force[2]=np.interp(disp[2],storey_disps,storey_forces)
              force[3]=np.interp(disp[3],storey_disps,storey_forces)

        # Trilinear
        elif len(storey_forces)==3:

              force[1]=storey_forces[0]
              force[4]=storey_forces[-1]

              disp[1]=storey_disps[0]
              disp[4]=storey_disps[-1]

              force[2]=storey_forces[1]
              disp[2] =storey_disps[1]

              disp[3]=np.mean([disp[2],disp[-1]])
              force[3]=np.interp(disp[3],storey_disps,storey_forces)

        # Quadrilinear
        elif len(storey_forces)==4:
              force[1]=storey_forces[0]
              force[4]=storey_forces[-1]

              disp[1]=storey_disps[0]
              disp[4]=storey_disps[-1]

              force[2]=storey_forces[1]
              disp[2]=storey_disps[1]

              force[3]=storey_forces[2]
              disp[3]=storey_disps[2]

        if degradation==True:
            matargs=[force[1,0],disp[1,0],force[2,0],disp[2,0],force[3,0],disp[3,0],force[4,0],disp[4,0],
                                 -1*force[1,0],-1*disp[1,0],-1*force[2,0],-1*disp[2,0],-1*force[3,0],-1*disp[3,0],-1*force[4,0],-1*disp[4,0],
                                 0.5,0.25,0.05,
                                 0.5,0.25,0.05,
                                 0,0.1,0,0,0.2,
                                 0,0.1,0,0,0.2,
                                 0,0.4,0,0.4,0.9,
                                 10,'energy']
        else:
            matargs=[force[1,0],disp[1,0],force[2,0],disp[2,0],force[3,0],disp[3,0],force[4,0],disp[4,0],
                                 -1*force[1,0],-1*disp[1,0],-1*force[2,0],-1*disp[2,0],-1*force[3,0],-1*disp[3,0],-1*force[4,0],-1*disp[4,0],
                                 0.5,0.25,0.05,
                                 0.5,0.25,0.05,
                                 0,0,0,0,0,
                                 0,0,0,0,0,
                                 0,0,0,0,0,
                                 10,'energy']

        ops.uniaxialMaterial('Pinching4', mat1Tag,*matargs)
        ops.uniaxialMaterial('MinMax', mat2Tag, mat1Tag, '-min', -1*disp[-1,0], '-max', disp[-1,0])

    def _plot_modes(self, node_list, mode_shape_vectors, T, export_path=None):
        """
        Plots the undeformed structure (3D, left) and 2D mode shape profiles (right).

        - 3D plot X and Y limits are set to encompass the structure coordinates and a minimum range of [-2, 2].
        - 3D plot Z-limit is fixed to start at 0.0.
        - Mode shape vectors are normalized by the X-displacement of the top node (max(Z)).

        Parameters:
            node_list (list): List of node tags.
            mode_shape_vectors (list of numpy.ndarray): Mode shape vectors (one per mode).
            T (list): List of natural periods corresponding to the modes.
            export_path (str, optional): If provided, saves the figure to this path
                                         (e.g., 'modes.png') instead of displaying it.
        """

        num_modes = len(T)

        # --- 1. Data Retrieval and Structuring ---
        node_coords_list = [ops.nodeCoord(tag) for tag in node_list]
        node_coords_undeformed = np.array(node_coords_list)
        element_list = ops.getEleTags()

        # Identify Base Nodes (first node) and Top Nodes (max Z coordinate)
        base_node_tag = node_list[0] if node_list else -1

        X, Y, Z = node_coords_undeformed.T
        z_max = np.max(Z)
        top_node_indices = np.where(Z == z_max)[0]

        # Z-levels for 2D plots (must be unique and ordered for interpolation)
        unique_z_levels = np.unique(Z)
        z_min = np.min(unique_z_levels)
        z_max = np.max(unique_z_levels)

        # --- CALCULATE 3D AXES LIMITS (Enforcing X/Y range [-2, 2] and Z min 0.0) ---
        x_min_data, x_max_data = np.min(X), np.max(X)
        y_min_data, y_max_data = np.min(Y), np.max(Y)
        z_min_data = np.min(Z)

        epsilon = 1e-6 # Small buffer for axes with zero extent

        # X Limits: Must span at least [-2, 2] and cover the data range plus a buffer
        x_min_3d = min(x_min_data, -2.0)
        x_max_3d = max(x_max_data, 2.0)
        x_range = x_max_3d - x_min_3d
        x_lim_3d = (x_min_3d - 0.05 * x_range, x_max_3d + 0.05 * x_range)
        if np.isclose(x_range, 0.0): # Handle case where all X coords are the same
            x_lim_3d = (x_min_3d - epsilon, x_max_3d + epsilon)

        # Y Limits: Must span at least [-2, 2] and cover the data range plus a buffer
        y_min_3d = min(y_min_data, -2.0)
        y_max_3d = max(y_max_data, 2.0)
        y_range = y_max_3d - y_min_3d
        y_lim_3d = (y_min_3d - 0.05 * y_range, y_max_3d + 0.05 * y_range)
        if np.isclose(y_range, 0.0): # Handle case where all Y coords are the same
            y_lim_3d = (y_min_3d - epsilon, y_max_3d + epsilon)

        # Z Limits: Force minimum to 0.0
        z_lim_3d = (max(0.0, z_min_data), z_max * 1.1)

        # --- 2. Create Figure and GridSpec Layout ---
        fig = plt.figure(figsize=(18, 10), facecolor='white')
        gs = GridSpec(num_modes, 3, figure=fig, width_ratios=[2, 0.1, 1], wspace=0.1)

        # --- TITLE ALIGNMENT: Set figure-wide title for 2D plots ---
        title_x_pos = 0.835 # Position centered over the right column
        fig.suptitle('2D Mode Shapes: Deformed Profile (X-Z View)',
                     fontsize=16,
                     weight='bold',
                     color='black',
                     y=0.95,
                     x=title_x_pos)

        # --- 3. Plot the 3D Axes (Left Side - UNDEFORMED ONLY) ---
        ax3d = fig.add_subplot(gs[:, 0], projection='3d')
        ax3d.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # 3D Aesthetics
        ax3d.set_xlabel('X-Direction [m]', fontsize=14, color='black', labelpad=10)
        ax3d.set_ylabel('Y-Direction [m]', fontsize=14, color='black', labelpad=10)
        ax3d.set_zlabel('Z-Direction [m]', fontsize=14, color='black', labelpad=10)
        ax3d.grid(True, linestyle=':', alpha=0.6, color='gray')
        ax3d.view_init(elev=20, azim=-60)

        # Set the corrected limits
        ax3d.set_xlim(x_lim_3d); ax3d.set_ylim(y_lim_3d); ax3d.set_zlim(z_lim_3d)
        ax3d.set_title('3D Undeformed Structure', fontsize=16, weight='bold', color='black', pad=15)

        # Plot Undeformed Nodes (Black markers)
        for i, node_tag in enumerate(node_list):
            x, y, z = node_coords_undeformed[i]
            marker_style = 's' if node_tag == base_node_tag else 'o'
            marker_size = 200 if node_tag == base_node_tag else 150
            ax3d.scatter(x, y, z, marker=marker_style, s=marker_size, color='black', zorder=2)

        # Plot Undeformed Elements (Solid Blue Line)
        for ele_tag in element_list:
            ele_nodes_tags = ops.eleNodes(ele_tag)
            if len(ele_nodes_tags) == 2:
                idx_i = node_list.index(ele_nodes_tags[0])
                idx_j = node_list.index(ele_nodes_tags[1])

                x_u = [node_coords_undeformed[idx_i, 0], node_coords_undeformed[idx_j, 0]]
                y_u = [node_coords_undeformed[idx_i, 1], node_coords_undeformed[idx_j, 1]]
                z_u = [node_coords_undeformed[idx_i, 2], node_coords_undeformed[idx_j, 2]]

                ax3d.plot(x_u, y_u, z_u, color='blue', linewidth=1.5, linestyle='-', alpha=0.7, zorder=1)

        # --- 4. Normalization and 2D Plot Setup ---
        normalized_mode_vectors = []
        for mode_vec in mode_shape_vectors:
            top_node_disp_x = mode_vec[top_node_indices, 0]
            max_top_disp = np.max(np.abs(top_node_disp_x))

            if max_top_disp != 0:
                normalized_vec = mode_vec / max_top_disp
            else:
                normalized_vec = mode_vec
            normalized_mode_vectors.append(normalized_vec)

        deformed_color = 'blue'
        max_disp_for_plotting = 1.0
        x_lim_2d = (-max_disp_for_plotting * 1.5, max_disp_for_plotting * 1.5)
        z_lim_2d = (z_min - 0.5, z_max + 0.5)

        # --- 5. Iterate through Modes for 2D Profile Plot (Right Side) ---

        for mode_idx, mode_vector in enumerate(normalized_mode_vectors):
            mode_num = mode_idx + 1
            period = T[mode_idx]

            ax2d = fig.add_subplot(gs[mode_idx, 2])

            # Extract 2D Plot Data (X-displacement vs Z-height)
            node_displacements_x = []
            for z_level in unique_z_levels:
                z_indices = np.where(Z == z_level)[0]
                node_displacements_x.append(np.mean(mode_vector[z_indices, 0]))

            # --- Interpolation Kind Selection ---
            N_z = len(unique_z_levels)
            if N_z < 3:
                interpolation_kind = 'linear'
            elif N_z == 3:
                interpolation_kind = 'quadratic'
            else:
                interpolation_kind = 'cubic'

            # 1. Undeformed Reference Line (Solid Gray Line)
            ax2d.plot([0] * N_z, unique_z_levels, color='gray', linewidth=3.0, linestyle='-', alpha=0.7, zorder=1)

            # 2. Plot Undeformed Nodes (Black Square/Circle at X=0)
            for i, node_tag in enumerate(node_list):
                z_u = node_coords_undeformed[i, 2]
                if z_u not in unique_z_levels: continue

                marker_style = 's' if node_tag == base_node_tag else 'o'
                marker_size = 80 if node_tag == base_node_tag else 50

                ax2d.scatter(0, z_u, marker=marker_style, s=marker_size, color='black', edgecolor='black', linewidth=0.5, zorder=2)

            # 3. Smooth Deformed Profile (Fixed Blue Line)
            f_interp = interp1d(unique_z_levels, node_displacements_x, kind=interpolation_kind)
            Z_smooth = np.linspace(z_min, z_max, 100)
            X_smooth = f_interp(Z_smooth)

            ax2d.plot(X_smooth, Z_smooth, color=deformed_color, linewidth=3.0, linestyle='-', zorder=4)

            # 4. Plot Deformed Nodes (Black Square/Circle at DISPLACED position)
            for i, node_tag in enumerate(node_list):
                z_u = node_coords_undeformed[i, 2]
                if z_u not in unique_z_levels: continue

                z_idx = np.where(unique_z_levels == z_u)[0][0]
                x_disp_at_z = node_displacements_x[z_idx]

                marker_style = 's' if node_tag == base_node_tag else 'o'
                marker_size = 80 if node_tag == base_node_tag else 50

                ax2d.scatter(x_disp_at_z, z_u, marker=marker_style, s=marker_size, color='black', edgecolor='black', linewidth=0.5, zorder=5)


            # 2D Plot Aesthetics and Labels
            title_text = f'Mode {mode_num}, $T_{{{mode_num}}} = {period:.3f}$ s'
            ax2d.set_title(title_text, fontsize=12, color='black', pad=5)

            ax2d.set_ylim(z_lim_2d)
            ax2d.grid(True, linestyle=':', alpha=0.5)
            ax2d.set_xlim(x_lim_2d)
            ax2d.set_ylabel('Z-Height [m]', fontsize=10)

            # X-Label placement
            if mode_idx < num_modes - 1:
                ax2d.tick_params(labelbottom=False)
                ax2d.set_xlabel(' ', fontsize=10)
            else:
                ax2d.set_xlabel('X-Displacement (Normalized)', fontsize=10)

            # Consistent Y-axis tick and label placement
            if mode_idx > 0:
                 ax2d.sharey(fig.axes[2])
                 ax2d.tick_params(labelleft=False)

        # Align labels again after tight_layout to finalize position
        fig.align_labels()
        plt.subplots_adjust(top=0.9)

        # --- FIGURE EXPORT/SHOW LOGIC ---
        if export_path:
            print(f"Saving figure to {export_path}")
            plt.savefig(export_path, bbox_inches='tight')
            plt.show()
        else:
            plt.show()

    # Helper function to create and save the animation for static pushover analysis
    def _animate_spo(self, spo_top_disp, spo_rxn, spo_disps, spo_midr, nodeList, elementList, push_dir, save_path):
        """Generates and saves the SPO animation using FuncAnimation."""
        deform_factor = 1 # Scaling factor for visualization
        # spo_midr is now passed in as an argument, so its length determines the number of frames.
        num_frames = len(spo_top_disp)

        # ------------------ Data Processing ------------------
        # Get undeformed coordinates once
        NodeCoordListX_und = [ops.nodeCoord(tag, 1) for tag in nodeList]
        NodeCoordListY_und = [ops.nodeCoord(tag, 2) for tag in nodeList]
        NodeCoordListZ_und = [ops.nodeCoord(tag, 3) for tag in nodeList]

        # Determine the plotting coordinates based on push_dir for the 2D view
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

        # Max coordinate for consistent plot limits
        max_abs_coord_x = np.max(np.abs(plot_coords_und[0]))
        max_abs_coord_y = np.max(np.abs(plot_coords_und[1]))
        model_x_lim = (-max_abs_coord_x * 3.0, max_abs_coord_x * 3.0)
        model_y_lim = (0, max_abs_coord_y * 1.5)

        # Max Interstorey Drift History (passed as spo_midr)
        max_drift_history = np.maximum.accumulate(spo_midr)

        # ------------------ Initialize the Figure and Subplots ------------------
        fig = plt.figure(figsize=(16, 8))

        # Layout: (1, 2, 1) is big left plot; (2, 2, 2) is top right; (2, 2, 4) is bottom right
        ax_model = fig.add_subplot(1, 2, 1)
        ax_curve = fig.add_subplot(2, 2, 2)
        ax_drift = fig.add_subplot(2, 2, 4)

        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # Store the number of static (undeformed) artists for easy cleanup in update()
        num_static_lines = len(elementList)
        num_static_collections = 1 # For the single undeformed nodes scatter plot

        # ------------------ Set up static plot elements ------------------
        # 2D Model Plot (Undeformed - static gray background)
        ax_model.scatter(plot_coords_und[0], plot_coords_und[1],
                          marker='o', s=50, color='gray', alpha=0.5, label='Undeformed Nodes')
        for eleTag in elementList:
            [NodeItag, NodeJtag] = ops.eleNodes(eleTag)
            i = nodeList.index(NodeItag)
            j = nodeList.index(NodeJtag)
            x_und = [plot_coords_und[0][i], plot_coords_und[0][j]]
            y_und = [plot_coords_und[1][i], plot_coords_und[1][j]]
            ax_model.plot(x_und, y_und, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        ax_model.set_xlabel(x_label_model)
        ax_model.set_ylabel(y_label_model)
        ax_model.set_title('Deformed Model Shape')
        ax_model.set_xlim(model_x_lim)
        ax_model.set_ylim(model_y_lim)
        ax_model.grid(True)

        # Pushover Curve (Base Shear vs Top Disp)
        ax_curve.set_xlabel('Top Displacement [m]')
        ax_curve.set_ylabel('Base Shear [kN]')
        ax_curve.set_title('Pushover Curve (Base Shear vs Top Disp)')
        ax_curve.plot(spo_top_disp, spo_rxn, 'gray', linewidth=2, alpha=0.5, label='Static Curve')
        curve_anim, = ax_curve.plot([], [], 'blue', linewidth=2, label='Current Step')
        ax_curve.legend()
        ax_curve.set_xlim(np.min(spo_top_disp)*1.1 if np.min(spo_top_disp) < 0 else 0, np.max(spo_top_disp)*1.1)
        ax_curve.set_ylim(np.min(spo_rxn)*1.1, np.max(spo_rxn)*1.1)
        ax_curve.grid(True)

        # Max Drift vs Base Shear
        ax_drift.set_xlabel('Max Interstorey Drift Ratio [%]')
        ax_drift.set_ylabel('Base Shear [kN]')
        ax_drift.set_title('Base Shear vs Max Interstorey Drift Ratio')
        drift_anim, = ax_drift.plot([], [], 'green', linewidth=2, label='Current Max Drift')
        ax_drift.legend()
        # Use spo_midr limits
        ax_drift.set_xlim(0, np.max(max_drift_history) * 1.2)
        ax_drift.set_ylim(np.min(spo_rxn)*1.1, np.max(spo_rxn)*1.1)
        ax_drift.grid(True)


        # ------------------ The update function for FuncAnimation ------------------
        def update(frame):
            nonlocal num_static_lines, num_static_collections

            # --- 2D Model Plot Cleanup ---
            # Remove dynamically drawn lines (deformed elements) from the LAST frame
            while len(ax_model.lines) > num_static_lines:
                ax_model.lines[-1].remove()

            # Remove dynamically drawn collections (deformed nodes) from the LAST frame
            while len(ax_model.collections) > num_static_collections:
                ax_model.collections[-1].remove()

            # --- 2D Model Plot Redraw (Deformed Shape) ---

            # Get displacement data for the current frame
            current_disps_floor = spo_disps[frame]
            # Include ground floor (index 0) displacement = 0
            full_node_disps = np.insert(current_disps_floor, 0, 0, axis=0)

            # Calculate deformed coordinates based on push_dir
            if push_dir == 1: # X-Z plane
                X_def = [plot_coords_und[0][i] + full_node_disps[i] * deform_factor for i in range(len(nodeList))]
                Z_def = [plot_coords_und[1][i] for i in range(len(nodeList))]
                plot_coords_def = (X_def, Z_def)
            elif push_dir == 2: # Y-Z plane
                Y_def = [plot_coords_und[0][i] + full_node_disps[i] * deform_factor for i in range(len(nodeList))]
                Z_def = [plot_coords_und[1][i] for i in range(len(nodeList))]
                plot_coords_def = (Y_def, Z_def)
            elif push_dir == 3: # Z-X plane
                Z_def = [plot_coords_und[0][i] + full_node_disps[i] * deform_factor for i in range(len(nodeList))]
                X_def = [plot_coords_und[1][i] for i in range(len(nodeList))]
                plot_coords_def = (Z_def, X_def)
            else:
                 plot_coords_def = plot_coords_und

            # Plot Deformed Shape (Blue)
            ax_model.scatter(plot_coords_def[0], plot_coords_def[1],
                              marker='o', s=50, color='blue', label='Deformed Nodes')
            for eleTag in elementList:
                [NodeItag, NodeJtag] = ops.eleNodes(eleTag)
                i = nodeList.index(NodeItag)
                j = nodeList.index(NodeJtag)
                x_def = [plot_coords_def[0][i], plot_coords_def[0][j]]
                y_def = [plot_coords_def[1][i], plot_coords_def[1][j]]
                ax_model.plot(x_def, y_def, color='blue', linewidth=1.5)

            ax_model.set_title(f'Frame: {frame}/{num_frames-1} (Scale: {deform_factor}x)')

            # --- Pushover Curve Update ---
            curve_anim.set_data(spo_top_disp[:frame+1], spo_rxn[:frame+1])

            # --- Max Drift vs Base Shear Update (Using spo_midr) ---
            drift_anim.set_data(spo_midr[:frame+1], spo_rxn[:frame+1])

            # Return the artists that were modified
            return curve_anim, drift_anim

        # Create the animation object
        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

        # Save the animation
        print(f"\nSaving animation to: {save_path}")

        if save_path.lower().endswith('.gif'):
            ani.save(save_path, writer='pillow', dpi=150)
        elif save_path.lower().endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', dpi=200)
        else:
            print("WARNING: Animation path extension not recognized (.gif or .mp4 recommended). Saving as default.")
            ani.save(save_path, dpi=150)

        plt.close(fig)

    # Helper function to create and save the animation for cyclic pushover analysis
    def _animate_cpo(self, cpo_dict, nodeList, elementList, push_dir, save_path):
        """
        Generates and saves the CPO animation using FuncAnimation, showing:
        1. Deformed model shape.
        2. Base shear vs. top displacement (hysteretic curve, spanning negative/positive).
        3. Base shear vs. maximum interstorey drift (newly updated to show hysteresis).

        Parameters
        ----------
        cpo_dict: dict
            The analysis results dictionary returned by do_cpo_analysis.
        nodeList: list
            List of node tags in the model.
        elementList: list
            List of element tags in the model.
        push_dir: int
            Direction of the pushover analysis (1=X, 2=Y, 3=Z).
        save_path: str
            File path to save the animation (e.g., 'cpo_animation.gif' or 'cpo_animation.mp4').
        """

        # ------------------ Data Extraction and Processing ------------------
        cpo_top_disp = cpo_dict['cpo_top_disp']
        cpo_rxn = cpo_dict['cpo_rxn']
        cpo_disps = cpo_dict['cpo_disps']
        cpo_drifts = cpo_dict['cpo_idr']

        deform_factor = 1.0 # Scaling factor for visualization
        num_frames = len(cpo_top_disp)

        # Calculate the maximum interstorey drift at each step:
        # Find the drift (with sign) of the floor that experiences the maximum absolute drift at this step.
        max_drift_indices = np.argmax(np.abs(cpo_drifts), axis=1)
        governing_drift_history = cpo_drifts[np.arange(num_frames), max_drift_indices]

        # Max absolute drift for setting limits
        max_drift_limit = np.max(np.abs(governing_drift_history))

        # Get undeformed coordinates once
        NodeCoordListX_und = [ops.nodeCoord(tag, 1) for tag in nodeList]
        NodeCoordListY_und = [ops.nodeCoord(tag, 2) for tag in nodeList]
        NodeCoordListZ_und = [ops.nodeCoord(tag, 3) for tag in nodeList]

        # Determine the plotting coordinates based on push_dir for the 2D view
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

        # Max coordinate for consistent plot limits
        max_abs_coord_x = np.max(np.abs(plot_coords_und[0]))
        max_abs_coord_y = np.max(np.abs(plot_coords_und[1]))
        model_x_lim = (-max_abs_coord_x * 3.0, max_abs_coord_x * 3.0)
        model_y_lim = (0, max_abs_coord_y * 1.5)

        # ------------------ Initialize the Figure and Subplots ------------------
        fig = plt.figure(figsize=(16, 8))

        # Layout: (1, 2, 1) is big left plot; (2, 2, 2) is top right; (2, 2, 4) is bottom right
        ax_model = fig.add_subplot(1, 2, 1)
        ax_curve = fig.add_subplot(2, 2, 2)
        ax_drift = fig.add_subplot(2, 2, 4)

        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # Store the number of static (undeformed) artists for easy cleanup in update()
        num_static_lines = len(elementList)
        num_static_collections = 1 # For the single undeformed nodes scatter plot

        # ------------------ Set up static plot elements (Undeformed Shape) ------------------
        ax_model.scatter(plot_coords_und[0], plot_coords_und[1],
                         marker='o', s=50, color='gray', alpha=0.5, label='Undeformed Nodes')
        for eleTag in elementList:
            try:
                [NodeItag, NodeJtag] = ops.eleNodes(eleTag)
                i = nodeList.index(NodeItag)
                j = nodeList.index(NodeJtag)
            except:
                continue

            x_und = [plot_coords_und[0][i], plot_coords_und[0][j]]
            y_und = [plot_coords_und[1][i], plot_coords_und[1][j]]
            ax_model.plot(x_und, y_und, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        ax_model.set_xlabel(x_label_model)
        ax_model.set_ylabel(y_label_model)
        ax_model.set_title('Deformed Model Shape (Cyclic Pushover)')
        ax_model.set_xlim(model_x_lim)
        ax_model.set_ylim(model_y_lim)
        ax_model.grid(True)

        # Hysteretic Curve (Base Shear vs Top Disp)
        ax_curve.set_xlabel('Top Displacement [m]')
        ax_curve.set_ylabel('Base Shear [kN]')
        ax_curve.set_title('Hysteretic Curve (Base Shear vs Top Disp)')
        ax_curve.plot(cpo_top_disp, cpo_rxn, 'gray', linewidth=1, alpha=0.5, label='History')
        curve_anim, = ax_curve.plot([], [], 'blue', linewidth=2, label='Current Step')
        ax_curve.legend(loc='lower right')

        # Set limits for cyclic analysis (must cover negative space)
        max_x_curve = np.max(np.abs(cpo_top_disp)) * 1.1
        max_y_curve = np.max(np.abs(cpo_rxn)) * 1.1
        ax_curve.set_xlim(-max_x_curve, max_x_curve)
        ax_curve.set_ylim(-max_y_curve, max_y_curve)
        ax_curve.grid(True)

        # Governing Drift Hysteresis (Base Shear vs MIDR) - UPDATED
        ax_drift.set_xlabel('Maximum Interstorey Drift [-]')
        ax_drift.set_ylabel('Base Shear [kN]')
        ax_drift.set_title('Hysteretic Curve (Base Shear vs MIDR)')
        # Plot full history in gray
        ax_drift.plot(governing_drift_history, cpo_rxn, 'gray', linewidth=1, alpha=0.5, label='History')
        # Plot current step in green
        drift_anim, = ax_drift.plot([], [], 'green', linewidth=2, label='Current Step')
        ax_drift.legend(loc='lower right')

        # Set limits for cyclic analysis (must cover negative space for drift) - UPDATED
        ax_drift.set_xlim(-max_drift_limit * 1.1, max_drift_limit * 1.1)
        ax_drift.set_ylim(-max_y_curve, max_y_curve)
        ax_drift.grid(True)


        # ------------------ The update function for FuncAnimation ------------------
        def update(frame):
            nonlocal num_static_lines, num_static_collections

            # --- 2D Model Plot Cleanup ---
            while len(ax_model.lines) > num_static_lines:
                ax_model.lines[-1].remove()

            while len(ax_model.collections) > num_static_collections:
                ax_model.collections[-1].remove()

            # --- 2D Model Plot Redraw (Deformed Shape) ---
            current_disps_floor = cpo_disps[frame]
            # Include ground floor (index 0) displacement = 0
            full_node_disps = np.insert(current_disps_floor, 0, 0, axis=0)

            # Calculate deformed coordinates based on push_dir
            if push_dir == 1: # X-Z plane
                X_def = [plot_coords_und[0][i] + full_node_disps[i] * deform_factor for i in range(len(nodeList))]
                Z_def = [plot_coords_und[1][i] for i in range(len(nodeList))]
                plot_coords_def = (X_def, Z_def)
            elif push_dir == 2: # Y-Z plane
                Y_def = [plot_coords_und[0][i] + full_node_disps[i] * deform_factor for i in range(len(nodeList))]
                Z_def = [plot_coords_und[1][i] for i in range(len(nodeList))]
                plot_coords_def = (Y_def, Z_def)
            elif push_dir == 3: # Z-X plane
                Z_def = [plot_coords_und[0][i] + full_node_disps[i] * deform_factor for i in range(len(nodeList))]
                X_def = [plot_coords_und[1][i] for i in range(len(nodeList))]
                plot_coords_def = (Z_def, X_def)
            else:
                plot_coords_def = plot_coords_und

            # Plot Deformed Shape (Blue)
            ax_model.scatter(plot_coords_def[0], plot_coords_def[1],
                             marker='o', s=50, color='blue', label='Deformed Nodes')
            for eleTag in elementList:
                try:
                    [NodeItag, NodeJtag] = ops.eleNodes(eleTag)
                    i = nodeList.index(NodeItag)
                    j = nodeList.index(NodeJtag)
                except:
                    continue

                x_def = [plot_coords_def[0][i], plot_coords_def[0][j]]
                y_def = [plot_coords_def[1][i], plot_coords_def[1][j]]
                ax_model.plot(x_def, y_def, color='blue', linewidth=1.5)

            ax_model.set_title(f'Frame: {frame}/{num_frames-1} (Scale: {deform_factor}x)')

            # --- Hysteretic Curve Update (Top Disp) ---
            curve_anim.set_data(cpo_top_disp[:frame+1], cpo_rxn[:frame+1])

            # --- Governing Drift Hysteresis Update ---
            drift_anim.set_data(governing_drift_history[:frame+1], cpo_rxn[:frame+1])

            # Return the artists that were modified
            return curve_anim, drift_anim

        # Create the animation object
        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

        # Save the animation
        print(f"\nSaving animation to: {save_path}")

        if save_path.lower().endswith('.gif'):
            ani.save(save_path, writer='pillow', dpi=300)
        elif save_path.lower().endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', dpi=300)
        else:
            print("WARNING: Animation path extension not recognized (.gif or .mp4 recommended). Saving as default.")
            ani.save(save_path, dpi=300)

        plt.close(fig)


    def compile_model(self):
        """
        Compiles and sets up the multi-degree-of-freedom (MDOF) oscillator model in OpenSees.

        This method constructs the model by defining nodes, assigning masses, imposing boundary conditions,
        and creating elements with associated material models for each storey in the building structure.
        It also defines rigid elastic materials for restrained degrees of freedom and nonlinear materials
        for unrestrained degrees of freedom. The method finally assembles the model for dynamic analysis.

        The process involves:
        1. Initializing the OpenSees model.
        2. Creating base and floor nodes.
        3. Assigning masses and degrees of freedom.
        4. Applying boundary conditions for the nodes.
        5. Creating zero-length elements for each storey with their respective material properties.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        - The method uses OpenSees' `ops.node`, `ops.mass`, and `ops.element` to define nodes, masses,
          and zero-length elements for the MDOF oscillator.
        - Boundary conditions are applied with the base node being fully fixed, while the upper storeys
          have horizontal degrees of freedom released.
        - The material model used for each storey is a Pinching4 hysteretic model, created by the
          `create_Pinching4_material` method.
        """

        ### Set model builder
        ops.wipe() # wipe existing model
        ops.model('basic', '-ndm', 3, '-ndf', 6)

        ### Define base node (tag = 0)
        ops.node(0, *[0.0, 0.0, 0.0])
        ### Define floor nodes (tag = 1+)
        i = 1
        current_height = 0.0
        while i <= self.number_storeys:
            nodeTag = i
            current_height = current_height + self.floor_heights[i-1]
            current_mass = self.floor_masses[i-1]
            coords = [0.0, 0.0, current_height]
            masses = [current_mass, current_mass, 1e-6, 1e-6, 1e-6, 1e-6]
            ops.node(nodeTag,*coords)
            ops.mass(nodeTag,*masses)
            i+=1

        ### Get list of model nodes
        nodeList = ops.getNodeTags()
        ### Impose boundary conditions
        for i in nodeList:
            # fix the base node against all DOFs
            if i==0:
                ops.fix(i,1,1,1,1,1,1)
            # release the horizontal DOFs (1,2) and fix remaining
            else:
                ops.fix(i,0,0,1,1,1,1)

        ### Get number of zerolength elements required
        nodeList = ops.getNodeTags()

        for i in range(self.number_storeys):

            ### define the material tag associated with each storey
            mat1Tag = int(f'1{i}00') # hysteretic material tag
            mat2Tag = int(f'1{i}01') # min-max material tag

            ### get the backbone curve definition
            current_storey_disps = self.storey_disps[i,:].tolist() # deformation capacity (i.e., storey displacement in m)
            current_storey_forces = self.storey_forces[i,:].tolist() # strength capacity (i.e., storey base shear in kN)

            ### Create rigid elastic materials for the restrained dofs
            rigM = int(f'1{i}02')
            ops.uniaxialMaterial('Elastic', rigM, 1e16)

            ### Create the nonlinear material for the unrestrained dofs
            self.create_Pinching4_material(mat1Tag, mat2Tag, current_storey_forces, current_storey_disps, self.degradation)

            ### Define element connectivity
            eleTag = int(f'200{i}')
            eleNodes = [i, i+1]

            ### Create the element
            ops.element('zeroLength', eleTag, eleNodes[0], eleNodes[1], '-mat', mat2Tag, mat2Tag, rigM, rigM, rigM, rigM, '-dir', 1, 2, 3, 4, 5, 6, '-doRayleigh', 1)

    def plot_model(self, display_info=True, export_path=None):
            """
            Plots the 3D visualization of the OpenSees model, including nodes and elements.

            This method generates a 3D plot of the multi-degree-of-freedom oscillator model defined in OpenSees.
            It visualizes the nodes and the connections between them (representing structural elements). Nodes
            are plotted as either square (base) or circular markers, while the elements are visualized as lines
            connecting the nodes. If `display_info` is set to True, the node coordinates and IDs will be displayed
            on the plot.

            Parameters
            ----------
            display_info : bool, optional
                If True, displays additional information (coordinates and node ID) next to each node in the plot.
                The default is True.
            export_path : str, optional
                If a string path (including filename and extension, e.g., 'model_plot.png') is provided,
                the plot is saved to this location instead of being displayed interactively.
                The default is None.

            Returns
            -------
            None

            Notes
            -----
            - Nodes are represented as either squares (base node) or circles (upper storey nodes).
            - Elements (connections between nodes) are represented by blue lines connecting the corresponding nodes.
            - Node coordinates are retrieved from OpenSees using `ops.nodeCoord` and node masses are retrieved with
              `ops.nodeMass`.
            - Element connectivity (pairs of nodes connected by an element) is retrieved using `ops.eleNodes`.
            - The plot is created using Matplotlib's 3D plotting functionality.
            """

            # get list of model nodes
            NodeCoordListX = []; NodeCoordListY = []; NodeCoordListZ = [];
            NodeMassList = []

            nodeList = ops.getNodeTags()
            for thisNodeTag in nodeList:
                NodeCoordListX.append(ops.nodeCoord(thisNodeTag,1))
                NodeCoordListY.append(ops.nodeCoord(thisNodeTag,2))
                NodeCoordListZ.append(ops.nodeCoord(thisNodeTag,3))
                NodeMassList.append(ops.nodeMass(thisNodeTag,1))

            # get list of model elements
            elementList = ops.getEleTags()
            for thisEleTag in elementList:
                eleNodesList = ops.eleNodes(thisEleTag)
                if len(eleNodesList)==2:
                    [NodeItag,NodeJtag] = eleNodesList
                    NodeCoordListI=ops.nodeCoord(NodeItag)
                    NodeCoordListJ=ops.nodeCoord(NodeJtag)
                    [NodeIxcoord,NodeIycoord,NodeIzcoord]=NodeCoordListI
                    [NodeJxcoord,NodeJycoord,NodeJzcoord]=NodeCoordListJ

            fig = plt.figure(figsize=(12,12))
            ax = fig.add_subplot(projection='3d')

            for i in range(len(nodeList)):
                if i==0:
                    ax.scatter(NodeCoordListX[i],NodeCoordListY[i],NodeCoordListZ[i], marker='s', s=200,color='black')
                else:
                    ax.scatter(NodeCoordListX[i],NodeCoordListY[i],NodeCoordListZ[i], marker='o', s=150,color='black')
                if display_info == True:
                    ax.text(NodeCoordListX[i]+0.01,NodeCoordListY[i],NodeCoordListZ[i],  'Node %s (%s,%s,%s)' % (str(i),str(NodeCoordListX[i]),str(NodeCoordListY[i]),str(NodeCoordListZ[i])), size=20, zorder=1, color="#0A4F5E")

            i = 0
            while i < len(elementList):

                x = [NodeCoordListX[i], NodeCoordListX[i+1]]
                y = [NodeCoordListY[i], NodeCoordListY[i+1]]
                z = [NodeCoordListZ[i], NodeCoordListZ[i+1]]

                plt.plot(x,y,z,color='blue')
                i = i+1

            ax.set_xlabel('X-Direction [m]', fontsize=14)
            ax.set_ylabel('Y-Direction [m]', fontsize=14)
            ax.set_zlabel('Z-Direction [m]', fontsize=14)

            # ----------------------------------------------------------------------
            # ADDED LOGIC FOR EXPORT_PATH
            # ----------------------------------------------------------------------
            if export_path:
                plt.savefig(export_path)
                plt.show()
            else:
                plt.show()

##########################################################################
#                             ANALYSIS MODULES                           #
##########################################################################
    def do_gravity_analysis(self, nG=100,
                            ansys_soe='UmfPack',
                            constraints_handler='Transformation',
                            numberer='RCM',
                            test_type='NormDispIncr',
                            init_tol = 1.0e-6,
                            init_iter = 500,
                            algorithm_type='Newton' ,
                            integrator='LoadControl',
                            analysis='Static'):
        """
        Perform a gravity analysis on a multi-degree-of-freedom (MDOF) system in OpenSees.

        This method sets up and runs a gravity analysis using specified parameters for various analysis objects
        in OpenSees. The gravity analysis solves for the static equilibrium of the system under self-weight loads
        (e.g., gravity loads).

        Parameters
        ----------
        nG: int, optional
            Number of gravity analysis steps to perform. Default is 100.

        ansys_soe: string, optional
            The system of equations type to be used in the analysis. This defines how the system of equations
            will be solved. Default is 'UmfPack' (sparse direct solver).

        constraints_handler: string, optional
            The constraints handler determines how the constraint equations are enforced in the analysis.
            It controls the enforcement of specified values for degrees-of-freedom (DOFs) or relationships
            between them. Default is 'Transformation' (transforming the constrained DOFs into active ones).

        numberer: string, optional
            The degree-of-freedom numberer defines how DOFs are numbered. This is important for system
            efficiency in solving. Default is 'RCM' (Reverse Cuthill-McKee, a reordering algorithm).

        test_type: string, optional
            Defines the test type used to check the convergence of the solution. It is used in constructing
            the LinearSOE and LinearSolver objects. Default is 'NormDispIncr' (norm of displacement increment).

        init_tol: float, optional
            The tolerance criterion for checking convergence. A smaller value means stricter convergence.
            Default is 1.0e-6.

        init_iter: int, optional
            The maximum number of iterations to check for convergence. Default is 500.

        algorithm_type: string, optional
            Defines the solution algorithm used in the analysis. Common options are 'Newton' (Newton-Raphson)
            for solving the system of equations. Default is 'Newton'.

        integrator: string, optional
            Defines the integrator for the analysis. The integrator dictates how the analysis steps are taken
            in time or load. Default is 'LoadControl' (control load increments).

        analysis: string, optional
            Defines the type of analysis to be performed. 'Static' is typically used for gravity analysis,
            but other options (e.g., 'Transient') can be used depending on the type of analysis. Default is 'Static'.

        Returns
        -------
        None.

        Notes
        -----
        - This method sets up the analysis using OpenSees by defining the system of equations, constraints
          handler, numberer, convergence test, solution algorithm, integrator, and analysis type.
        - The gravity analysis solves for the static equilibrium under self-weight or gravity loads and is
          typically used to determine the initial equilibrium state of a structure before dynamic loading.
        - The analysis can be modified by changing the parameters to adjust solver settings, tolerance,
          and other relevant options.
        - After the analysis is completed, the analysis objects are wiped to ensure a clean state for further analyses.
        """

        ### Define the analysis objects and run gravity analysis
        ops.system(ansys_soe) # creates the system of equations, a sparse solver with partial pivoting
        ops.constraints(constraints_handler) # creates the constraint handler, the transformation method
        ops.numberer(numberer) # creates the DOF numberer, the reverse Cuthill-McKee algorithm
        ops.test(test_type, init_tol, init_iter, 3) # creates the convergence test
        ops.algorithm(algorithm_type) # creates the solution algorithm, a Newton-Raphson algorithm
        ops.integrator(integrator, (1/nG)) # creates the integration scheme
        ops.analysis(analysis) # creates the analysis object
        ops.analyze(nG) # perform the gravity load analysis
        ops.loadConst('-time', 0.0)

        ### Wipe the analysis objects
        ops.wipeAnalysis()

    def do_modal_analysis(self,
                          num_modes=3,
                          solver='-genBandArpack',
                          doRayleigh=False,
                          pflag=False,
                          plot_modes=True,
                          export_path=None): # NEW PARAMETER for export
        """
        Perform modal analysis on a multi-degree-of-freedom (MDOF) system and optionally plot/export the mode shapes.

        Parameters
        ----------
        num_modes: int, optional
            The number of modes to consider in the analysis. Default is 3.
        solver: string, optional
            The type of solver to use for the eigenvalue problem. Default is '-genBandArpack'.
        doRayleigh: bool, optional
            Flag to enable or disable Rayleigh damping. Default is False. (Not used directly here).
        pflag: bool, optional
            Flag to control whether to print the modal analysis report. Default is False.
        plot_modes: bool, optional
            If True, initiates plotting of the mode shapes. Default is True.
        export_path: str or None, optional
            If a string path is provided (e.g., 'modal_results.png'), the plot will be saved to this location.
            If None, the plot will be displayed. Default is None.

        Returns
        -------
        T: array
            The periods of vibration for the system.
        mode_shape_vectors: list
            A list of numpy arrays, where each array is the full mode shape vector (UX, UY, UZ) for all nodes.
        """

        # --- 1. Perform Modal Analysis (Eigenvalue Problem) ---
        self.omega = np.power(ops.eigen(solver, num_modes), 0.5)
        T = 2.0 * np.pi / self.omega

        # --- 2. Extract Mode Shape Vectors ---
        node_list = ops.getNodeTags()

        # Fallback/Adaptive: Determine the largest node tag index for eigenvector extraction
        if not hasattr(self, 'number_storeys'):
            self.number_storeys = len(node_list)

        mode_shape_vectors = []

        for mode_num in range(1, num_modes + 1):
            # Extract X, Y, Z displacements for ALL nodes in the current mode
            ux_all = np.array([ops.nodeEigenvector(tag, mode_num, 1) for tag in node_list])
            uy_all = np.array([ops.nodeEigenvector(tag, mode_num, 2) for tag in node_list])
            uz_all = np.array([ops.nodeEigenvector(tag, mode_num, 3) for tag in node_list])

            # Combine into a single (N_nodes x 3) vector for plotting
            mode_vector = np.column_stack((ux_all, uy_all, uz_all))

            # Normalization (This normalization is overridden inside _plot_modes
            # to ensure it's normalized by top node X-disp, but kept here for return consistency)
            max_disp = np.max(np.abs(mode_vector))
            if max_disp != 0:
                mode_vector /= max_disp

            mode_shape_vectors.append(mode_vector)

        # --- 3. Optional Printing ---
        if pflag:
            ops.modalProperties('-print')
            print(r'Fundamental Period: T = {:.3f} s'.format(T[0]))

        # --- 4. Optional Plotting/Exporting ---
        if plot_modes:
            # Pass the export_path to the plotting function
            self._plot_modes(node_list, mode_shape_vectors, T, export_path=export_path)

        # --- 5. Cleanup ---
        ops.wipeAnalysis()

        return T, mode_shape_vectors

    def do_spo_analysis(self,
                        ref_disp,
                        disp_scale_factor,
                        push_dir,
                        phi,
                        pflag=True,
                        num_steps=200,
                        ansys_soe='BandGeneral',
                        constraints_handler='Transformation',
                        numberer='RCM',
                        test_type='EnergyIncr',
                        init_tol=1.0e-5,
                        init_iter=1000,
                        algorithm_type='KrylovNewton',
                        save_animation_path=None):
        """
        Perform static pushover analysis (SPO) on a multi-degree-of-freedom (MDOF) system.

        This method simulates a static pushover analysis where a lateral load pattern is incrementally applied
        to the structure. The displacement at the control node is increased step by step, and the corresponding
        base shear, floor displacements, and forces in non-linear elements are recorded. The analysis helps in
        evaluating the structural response to lateral loads, such as earthquake forces.

        Parameters
        ----------
        ref_disp: float
            The reference displacement at which the analysis starts, corresponding to the yield or other
            significant displacement (e.g., 1mm).

        disp_scale_factor: float
            The scale factor applied to the reference displacement to determine the final displacement.
            The analysis will be run to this scaled displacement.

        push_dir: int
            The direction in which the pushover load is applied:
                1 = X direction
                2 = Y direction
                3 = Z direction

        phi: list of floats
            The lateral load pattern shape. This is typically a mode shape or a predefined load distribution.
            For example, it can be the first-mode shape from the calibrateModel function.

        pflag: bool, optional
            Flag to print (or not) the pushover analysis steps. If True, detailed feedback on each step will be printed. Default is True.

        num_steps: int, optional
            The number of steps to increment the pushover load. Default is 200.

        ansys_soe: string, optional
            The type of system of equations solver to use. Default is 'BandGeneral'.

        constraints_handler: string, optional
            The constraints handler object to determine how constraint equations are enforced. Default is 'Transformation'.

        numberer: string, optional
            The degree-of-freedom (DOF) numberer object to determine the mapping between equation numbers and degrees-of-freedom. Default is 'RCM'.

        test_type: string, optional
            The type of test to use for the linear system of equations. Default is 'EnergyIncr'.

        init_tol: float, optional
            The tolerance criterion to check for convergence. Default is 1.0e-5.

        init_iter: int, optional
            The maximum number of iterations to perform when checking for convergence. Default is 1000.

        algorithm_type: string, optional
            The type of algorithm used to solve the system. Default is 'KrylovNewton'.

        Returns
        -------
        spo_dict: dict
            A dictionary containing the SPO results with the following keys:
            'spo_disps': array - Displacements at each floor level (TimeSteps x Floors).
            'spo_rxn': array - Base shear recorded at the base (TimeSteps).
            'spo_disps_spring': array - Displacements in the storey zero-length elements (TimeSteps x Springs).
            'spo_forces_spring': array - Shear forces in the storey zero-length elements (TimeSteps x Springs).
            'spo_idr': array - Interstorey drift ratio history for each storey (TimeSteps x Storeys).
            'spo_midr': array - Maximum interstorey drift ratio history (max IDR across all stories at each step, TimeSteps).
        """

        # --- Setup OpenSees Model for Analysis ---
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)

        nodeList = ops.getNodeTags()
        control_node = nodeList[-1]
        pattern_nodes = nodeList[1:]
        rxn_nodes = [nodeList[0]] # Base node for reaction calculation

        # Apply the lateral load pattern
        for i in np.arange(len(pattern_nodes)):
            load_val = 1.0 if len(pattern_nodes)==1 else phi[i]*self.floor_masses[i]
            if push_dir == 1:
                ops.load(pattern_nodes[i], load_val, 0.0, 0.0, 0.0, 0.0, 0.0)
            elif push_dir == 2:
                ops.load(pattern_nodes[i], 0.0, load_val, 0.0, 0.0, 0.0, 0.0)
            elif push_dir == 3:
                ops.load(pattern_nodes[i], 0.0, 0.0, load_val, 0.0, 0.0, 0.0)

        # Set analysis objects
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)
        ops.algorithm(algorithm_type)

        # Set integrator
        target_disp = float(ref_disp)*float(disp_scale_factor)
        delta_disp = target_disp/(1.0*num_steps)
        ops.integrator('DisplacementControl', control_node, push_dir, delta_disp)
        ops.analysis('Static')

        elementList = ops.getEleTags()

        if pflag is True:
            print(f"\n------ Static Pushover Analysis of Node # {control_node} to {target_disp} ---------")

        ok = 0
        step = 1
        loadf = 1.0

        # Initialize result arrays with current state (usually 0.0)
        spo_rxn = np.array([0.])
        spo_top_disp = np.array([ops.nodeResponse(control_node, push_dir,1)]) # Used for animation and Pushover Curve
        spo_disps = np.array([[ops.nodeResponse(node, push_dir, 1) for node in pattern_nodes]])
        spo_disps_spring = np.array([[ops.eleResponse(ele, 'deformation')[0] for ele in elementList]])
        spo_forces_spring = np.array([[ops.eleResponse(ele, 'force')[0] for ele in elementList]])

        # --- Main Analysis Loop ---
        while step <= num_steps and ok == 0 and loadf > 0:

            ok = ops.analyze(1)

            # --- Adaptive Convergence Scheme ---
            if ok != 0:
                if pflag: print('FAILED: Trying relaxing convergence...')
                ops.test(test_type, init_tol*0.01, init_iter)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                if pflag: print('FAILED: Trying relaxing convergence with more iterations...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                if pflag: print('FAILED: Trying relaxing convergence with more iteration and Newton with initial then current...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ops.algorithm('Newton', 'initialThenCurrent')
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                if pflag: print('FAILED: Trying relaxing convergence with more iteration and Newton with initial...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ops.algorithm('Newton', 'initial')
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                if pflag: print('FAILED: Attempting a Hail Mary...')
                ops.test('FixedNumIter', init_iter*10)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
                if ok != 0: # Final check before breaking
                    break

            loadf = ops.getTime()

            if pflag is True:
                curr_disp = ops.nodeDisp(control_node, push_dir)
                print(f'Currently pushed node {control_node} to {curr_disp:.4f} with load factor {loadf:.4f}')

            step += 1

            # --- Record Results ---
            spo_top_disp = np.append(spo_top_disp, ops.nodeResponse(control_node, push_dir, 1))

            current_disps = np.array([ops.nodeResponse(node, push_dir, 1) for node in pattern_nodes])
            spo_disps = np.append(spo_disps, np.array([current_disps]), axis=0)

            spo_disps_spring = np.append(spo_disps_spring, np.array([
                [ops.eleResponse(ele, 'deformation')[0] for ele in elementList]
            ]), axis=0)

            spo_forces_spring = np.append(spo_forces_spring, np.array([
                [ops.eleResponse(ele, 'force')[0] for ele in elementList]
            ]), axis=0)

            ops.reactions()
            temp = 0
            for n in rxn_nodes:
                temp += ops.nodeReaction(n, push_dir)
            spo_rxn = np.append(spo_rxn, -temp)


        # --- Final Cleanup and Output ---
        if ok != 0:
            print('------ ANALYSIS FAILED --------')
        elif ok == 0:
            print('~~~~~~~ ANALYSIS SUCCESSFUL ~~~~~~~~~')
        if loadf < 0:
            print('Stopped because of load factor below zero')

        ops.wipeAnalysis()

        # -----------------------------------------------------------------
        # 3. Calculate Interstorey Drift Ratio (IDR) and Max IDR (MIDR)
        # -----------------------------------------------------------------

        # Use a COPY of the original displacement history for IDR calculation
        idr_disps = spo_disps.copy()

        if not hasattr(self, 'floor_heights'):
            raise AttributeError("Cannot calculate IDR: 'floor_heights' property is required but not defined in the class.")

        # Step 1: Prepend ground floor (zero displacement)
        ground_disps = np.zeros((idr_disps.shape[0], 1))
        full_idr_disps = np.hstack([ground_disps, idr_disps])

        # Step 2: Compute interstorey displacements (ISD)
        spo_isd = np.diff(full_idr_disps, axis=1)

        # Convert floor_heights to a numpy array for division
        floor_heights = np.array(self.floor_heights)

        # Step 3: Normalize by corresponding floor heights to get IDR (x100 requested)
        spo_idr = (spo_isd / floor_heights) * 100

        # Step 4: Take the maximum interstorey drift ratio per step
        spo_midr = np.max(np.abs(spo_idr), axis=1)

        # 4. Handle Animation (Call updated function with spo_midr)
        if save_animation_path:
            self._animate_spo(spo_top_disp, spo_rxn, spo_disps, spo_midr, nodeList, elementList, push_dir, save_animation_path)

        # 5. Pack and Return results into a dictionary
        spo_dict = {'spo_disps': spo_disps,
                    'spo_rxn': spo_rxn,
                    'spo_disps_spring': spo_disps_spring,
                    'spo_forces_spring': spo_forces_spring,
                    'spo_idr': spo_idr,
                    'spo_midr': spo_midr}

        return spo_dict

    def do_cpo_analysis(self,
                        ref_disp,
                        mu_levels,
                        push_dir,
                        dispIncr,
                        phi,
                        pflag=True,
                        ansys_soe='BandGeneral',
                        constraints_handler='Transformation',
                        numberer='RCM',
                        test_type='NormDispIncr',
                        init_tol=1.0e-5,
                        init_iter=1000,
                        algorithm_type='KrylovNewton',
                        save_animation_path=None):
        """
        Perform cyclic pushover (CPO) analysis on a Multi-Degree-of-Freedom (MDOF) system.

        Parameters
        ----------
        ref_disp: float
            Reference displacement (e.g., yield displacement) for scaling the cycles.
        mu_levels: list
            Target ductility factors (mu) for each cycle level.
        push_dir: int
            Direction of the pushover analysis (1=X, 2=Y, 3=Z).
        dispIncr: int
            The number of displacement increments for each loading cycle target.
        phi: list of floats
            The lateral load pattern shape vector (scaled by mass).
        pflag: bool, optional, default=True
            If True, prints feedback during the analysis steps.
        save_animation_path: str, optional, default=None
            If provided, the path to save the animation (e.g., 'cpo.gif' or 'cpo.mp4').
        ansys_soe: string, optional, default='BandGeneral'
            System of equations solver.
        constraints_handler: string, optional, default='Transformation'
            Constraint handler method.
        numberer: string, optional, default='RCM'
            The numberer method.
        test_type: string, optional, default='NormDispIncr'
            Convergence test type.
        init_tol: float, optional, default=1e-5
            The initial tolerance for convergence.
        init_iter: int, optional, default=1000
            The maximum number of iterations for the solver.
        algorithm_type: string, optional, default='KrylovNewton'
            The type of algorithm used to solve the system of equations.

        Returns
        -------
        cpo_dict: dict
            A dictionary containing all the analysis results (displacements, base_shear, etc.).
        """

        if mu_levels is None:
            mu_levels = [1, 2, 4, 6, 8, 10]

        # Apply the load pattern
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain",1,1)

        # Get all tags needed for analysis and animation
        nodeList = ops.getNodeTags()
        elementList = ops.getEleTags()

        # Ensure model has nodes
        if not nodeList:
            print("ERROR: No nodes found in the OpenSees model.")
            return None

        control_node = nodeList[-1]
        pattern_nodes = nodeList[1:] # All nodes above ground
        rxn_nodes = [nodeList[0]] # Ground node

        # Quality control
        assert len(phi) == len(pattern_nodes), "phi length must match pattern_nodes"
        assert len(self.floor_masses) == len(pattern_nodes), "floor_masses length mismatch"

        # Apply lateral load pattern scaled by mass
        for i in np.arange(len(pattern_nodes)):
            if push_dir == 1:
                ops.load(pattern_nodes[i], phi[i]*self.floor_masses[i], 0.0, 0.0, 0.0, 0.0, 0.0)
            elif push_dir == 2:
                ops.load(pattern_nodes[i], 0.0, phi[i]*self.floor_masses[i], 0.0, 0.0, 0.0, 0.0)
            elif push_dir == 3:
                ops.load(pattern_nodes[i], 0.0, 0.0, phi[i]*self.floor_masses[i], 0.0, 0.0, 0.0)

        # Set up the analysis objects
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)
        ops.algorithm(algorithm_type)

        # Create the list of target displacements (e.g., +1mu, -1mu, +2mu, -2mu, ...)
        cycleDispList = []
        for mu in mu_levels:
            cycleDispList.append(ref_disp * mu)   # push positive
            cycleDispList.append(-ref_disp * mu)  # pull negative
        dispNoMax = len(cycleDispList)

        if pflag:
            print(f"\n------ Cyclic Pushover with ductility levels: {mu_levels} ------")

        # Recording data arrays
        cpo_rxn = [0.0]
        cpo_top_disp = [ops.nodeDisp(control_node, push_dir)]
        cpo_disps = [[ops.nodeDisp(node, push_dir) for node in pattern_nodes]]
        energy_steps = [0.0]

        for d in range(dispNoMax):
            numIncr = dispIncr
            current_disp = ops.nodeDisp(control_node, push_dir)
            target_disp = cycleDispList[d]
            dU = (target_disp - current_disp) / numIncr

            # Use DisplacementControl integrator
            ops.integrator('DisplacementControl', control_node, push_dir, dU)
            ops.analysis('Static')

            # Loop over displacement increments
            for l in range(numIncr):
                ok = ops.analyze(1)

                # --- Convergence Failure Handling (Extended Recovery) ---
                if ok != 0:
                    print(f'FAILED at cycle {d+1}/{dispNoMax}, increment {l}/{numIncr}: Starting complex recovery attempts...')

                # 1. Try relaxing convergence tolerance
                if ok != 0:
                    print('FAILED: Trying relaxing convergence...')
                    ops.test(test_type, init_tol*0.01, init_iter)
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)

                # 2. Try relaxing convergence tolerance with more iterations
                if ok != 0:
                    print('FAILED: Trying relaxing convergence with more iterations...')
                    ops.test(test_type, init_tol*0.01, init_iter*10)
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)

                # 3. Try relaxing tolerance, more iterations, and Newton with 'initialThenCurrent'
                if ok != 0:
                    print('FAILED: Trying relaxing convergence with more iteration and Newton with initial then current...')
                    ops.test(test_type, init_tol*0.01, init_iter*10)
                    ops.algorithm('Newton', 'initialThenCurrent')
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)
                    ops.algorithm(algorithm_type) # Restore original algorithm

                # 4. Try relaxing tolerance, more iterations, and Newton with 'initial'
                if ok != 0:
                    print('FAILED: Trying relaxing convergence with more iteration and Newton with initial...')
                    ops.test(test_type, init_tol*0.01, init_iter*10)
                    ops.algorithm('Newton', 'initial')
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)
                    ops.algorithm(algorithm_type) # Restore original algorithm

                # 5. Attempt a Hail Mary (FixedNumIter)
                if ok != 0:
                    print('FAILED: Attempting a Hail Mary...')
                    ops.test('FixedNumIter', init_iter*10)
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter) # Restore original test type

                # Final failure check
                if ok != 0:
                    print('Analysis Failed')
                    break

                # --- Data Recording (only if successful) ---
                if ok == 0:
                    curr_disp = ops.nodeDisp(control_node, push_dir)
                    cpo_top_disp.append(curr_disp)

                    current_floor_disps = [ops.nodeDisp(node, push_dir) for node in pattern_nodes]
                    cpo_disps.append(current_floor_disps)

                    ops.reactions()
                    temp = sum(ops.nodeReaction(n, push_dir) for n in rxn_nodes)
                    curr_rxn = -temp
                    cpo_rxn.append(curr_rxn)

                    if len(cpo_top_disp) >= 2:
                        dU_step = cpo_top_disp[-1] - cpo_top_disp[-2]
                        avg_F = 0.5 * (cpo_rxn[-1] + cpo_rxn[-2])
                        dE = abs(avg_F * dU_step)
                        energy_steps.append(energy_steps[-1] + dE)
                    else:
                        energy_steps.append(energy_steps[-1])

            if pflag is True:
                curr_disp = ops.nodeDisp(control_node, push_dir)
                print(f"Cycle target {d+1}/{dispNoMax}: Pushed node {control_node} to {curr_disp:.4f}")

        # Convert lists to numpy arrays
        cpo_rxn = np.array(cpo_rxn)
        cpo_top_disp = np.array(cpo_top_disp)
        cpo_disps = np.array(cpo_disps)
        pseudo_steps = np.arange(len(energy_steps))
        cpo_energy = np.column_stack((pseudo_steps, energy_steps))

        # --- Calculate Interstorey Drifts ---
        base_disps = np.zeros((cpo_disps.shape[0], 1))
        padded_disps = np.hstack((base_disps, cpo_disps))
        cpo_drifts = np.diff(padded_disps, axis=1)
        max_interstorey_drift = np.max(np.abs(cpo_drifts))
        # ------------------------------------

        ops.wipeAnalysis()

        # Final output dictionary (cpo_dict)
        cpo_dict = {'cpo_disps': cpo_disps,
                    'cpo_rxn': cpo_rxn,
                    'cpo_top_disp': cpo_top_disp,
                    'cpo_energy': cpo_energy,
                    'cpo_idr': cpo_drifts,
                    'cpo_midr': max_interstorey_drift}

        # ------------------ ANIMATION CALL ------------------
        if save_animation_path:
            self._animate_cpo(cpo_dict, nodeList, elementList, push_dir, save_animation_path)
        # ----------------------------------------------------

        return cpo_dict

    def do_nrha_analysis(self, fnames, dt_gm, sf, t_max, dt_ansys, nrha_outdir,
                         pflag=True, xi = 0.05, ansys_soe='BandGeneral',
                         constraints_handler='Plain', numberer='RCM',
                         test_type='NormDispIncr', init_tol=1.0e-6, init_iter=50,
                         algorithm_type='Newton'):
        """
        Perform nonlinear time-history analysis on a Multi-Degree-of-Freedom (MDOF) system.

        This method performs a nonlinear time-history analysis where ground motion records are applied to the
        system to simulate real-world seismic conditions. The analysis uses step-by-step integration methods
        to solve the system's response under dynamic loading.

        Parameters
        ----------
        fnames: list
            List of file paths to the ground motion records for each direction (X, Y, Z). At least one ground motion
            record in the X direction is required.

        dt_gm: float
            Time-step of the ground motion records, which is typically the time between each data point in the records.

        sf: float
            Scale factor to apply to the ground motion records. Typically equal to the gravitational acceleration (9.81 m/s²).

        t_max: float
            The maximum time duration for the analysis. It is typically the total time span of the ground motion record.

        dt_ansys: float
            The time-step at which the analysis will be conducted. Typically smaller than the ground motion time-step to
            ensure accurate results.

        nrha_outdir: string
            Directory where temporary output files (e.g., acceleration records) are saved during the analysis.

        pflag: bool, optional, default=True
            Flag to print progress updates during the analysis. If True, the function prints information about the analysis
            steps and progress.

        xi: float, optional, default=0.05
            The inherent damping ratio used in the analysis. The default is 5% damping (0.05).

        ansys_soe: string, optional, default='BandGeneral'
            Type of the system of equations solver to be used in the analysis (e.g., 'BandGeneral', 'FullGeneral', etc.).

        constraints_handler: string, optional, default='Plain'
            The method used to handle constraints in the analysis. This handles how boundary conditions or prescribed
            displacements are enforced.

        numberer: string, optional, default='RCM'
            The numberer object determines the equation numbering used in the analysis. Default is 'RCM' (Reverse Cuthill-McKee).

        test_type: string, optional, default='NormDispIncr'
            Type of convergence test used during the analysis to check whether the solution has converged. Default is 'NormDispIncr'.

        init_tol: float, optional, default=1.0e-6
            Initial tolerance for the convergence test, used to check if the solution is converging to a sufficiently accurate result.

        init_iter: int, optional, default=50
            Maximum number of iterations allowed during each time step for the analysis to converge.

        algorithm_type: string, optional, default='Newton'
            Type of algorithm used to solve the system of equations. Default is 'Newton', which uses the Newton-Raphson method.

        Returns
        -------
        control_nodes: list
            List of the floor node tags in the MDOF system.

        conv_index: int
            Convergence status index: -1 indicates failure, 0 indicates success (converged).

        peak_drift: numpy.ndarray
            Array of peak storey drift values for each storey in the X and Y directions (radians).

        peak_accel: numpy.ndarray
            Array of peak floor acceleration values for each floor in the X and Y directions (g).

        max_peak_drift: float
            The maximum peak storey drift value (radians) across all floors.

        max_peak_drift_dir: string
            Direction of the maximum peak storey drift ('X' or 'Y').

        max_peak_drift_loc: int
            Location (storey) of the maximum peak storey drift.

        max_peak_accel: float
            The maximum peak floor acceleration value (g) across all floors.

        max_peak_accel_dir: string
            Direction of the maximum peak floor acceleration ('X' or 'Y').

        max_peak_accel_loc: int
            Location (floor) of the maximum peak floor acceleration.

        peak_disp: numpy.ndarray
            Array of peak displacement values (in meters) for each floor.
        """

        # define control nodes
        control_nodes = ops.getNodeTags()

        # Define the timeseries and patterns first
        if len(fnames) > 0:
            nrha_tsTagX = 1
            nrha_pTagX = 1
            ops.timeSeries('Path', nrha_tsTagX, '-dt', dt_gm, '-filePath', fnames[0], '-factor', sf)
            ops.pattern('UniformExcitation', nrha_pTagX, 1, '-accel', nrha_tsTagX)
            ops.recorder('Node', '-file', f"{nrha_outdir}/floor_accel_X.txt", '-timeSeries', nrha_tsTagX, '-node', *control_nodes, '-dof', 1, 'accel')
        if len(fnames) > 1:
            nrha_tsTagY = 2
            nrha_pTagY = 2
            ops.timeSeries('Path', nrha_tsTagY, '-dt', dt_gm, '-filePath', fnames[1], '-factor', sf)
            ops.pattern('UniformExcitation', nrha_pTagY, 2, '-accel', nrha_tsTagY)
            ops.recorder('Node', '-file', f"{nrha_outdir}/floor_accel_Y.txt", '-timeSeries', nrha_tsTagY, '-node', *control_nodes, '-dof', 2, 'accel')
        if len(fnames) > 2:
            nrha_tsTagZ = 3
            nrha_pTagZ = 3
            ops.timeSeries('Path', nrha_tsTagZ, '-dt', dt_gm, '-filePath', fnames[2], '-factor', sf)
            ops.pattern('UniformExcitation', nrha_pTagZ, 3, '-accel', nrha_tsTagZ)

        # Set up the initial objects
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)
        ops.algorithm(algorithm_type)
        ops.integrator('Newmark', 0.5, 0.25)
        ops.analysis('Transient')

        # Set up analysis parameters
        conv_index = 0   # Initially define the collapse index (-1 for non-converged, 0 for stable)
        control_time = 0.0
        ok = 0 # Set the convergence to 0 (initially converged)

        # Parse the data about the building
        top_nodes = control_nodes[1:]
        bottom_nodes = control_nodes[0:-1]
        h = []
        for i in np.arange(len(top_nodes)):
            topZ = ops.nodeCoord(top_nodes[i], 3)
            bottomZ = ops.nodeCoord(bottom_nodes[i], 3)
            dist = topZ - bottomZ
            if dist == 0:
                print("WARNING: Zero length found in drift check, using very large distance 1e9 instead")
                h.append(1e9)
            else:
                h.append(dist)

        # Create some arrays to record to
        peak_disp = np.zeros((len(control_nodes), 2))
        peak_drift = np.zeros((len(top_nodes), 2))
        peak_accel = np.zeros((len(top_nodes)+1, 2))

        # Set damping
        if self.number_storeys == 1:

            #Set damping
            alphaM = 2*self.omega[0]*xi
            ops.rayleigh(alphaM,0,0,0)

        else:

            alphaM = 2*self.omega[0]*self.omega[2]*xi/(self.omega[0] + self.omega[2])
            alphaK = 2*xi/(self.omega[0] + self.omega[2])
            ops.rayleigh(alphaM,0,alphaK,0)

        # Define parameters for deformation animation
        # n_steps = int(np.ceil(t_max/dt_gm))+1
        # node_disps = np.zeros([n_steps,len(control_nodes)])
        # node_accels= np.zeros([n_steps,len(control_nodes)])

        # Run the actual analysis
        # step = 0 # initialise the step counter
        while conv_index == 0 and control_time <= t_max and ok == 0:
            ok = ops.analyze(1, dt_ansys)
            control_time = ops.getTime()

            if pflag is True:
                print('Completed {:.3f}'.format(control_time) + ' of {:.3f} seconds'.format(t_max) )

            # If the analysis fails, try the following changes to achieve convergence
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying reducing time-step in half...')
                ok = ops.analyze(1, 0.5*dt_ansys)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying reducing time-step in quarter...')
                ok = ops.analyze(1, 0.25*dt_ansys)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying relaxing convergence with more iterations...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ok = ops.analyze(1, 0.5*dt_ansys)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying relaxing convergence with more iteration and Newton with initial then current...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ops.algorithm('Newton', 'initialThenCurrent')
                ok = ops.analyze(1, 0.5*dt_ansys)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Trying relaxing convergence with more iteration and Newton with initial...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ops.algorithm('Newton', 'initial')
                ok = ops.analyze(1, 0.5*dt_ansys)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Attempting a Hail Mary...')
                ops.test('FixedNumIter', init_iter*10)
                ok = ops.analyze(1, 0.5*dt_ansys)
                ops.test(test_type, init_tol, init_iter)

            # Game over......
            if ok != 0:
                print('FAILED at {:.3f}'.format(control_time) + ': Exiting analysis...')
                conv_index = -1

            # For each of the nodes to monitor, get the current drift
            for i in np.arange(len(top_nodes)):

                # Get the current storey drifts - absolute difference in displacement over the height between them
                curr_drift_X = np.abs(ops.nodeDisp(top_nodes[i], 1) - ops.nodeDisp(bottom_nodes[i], 1))/h[i]
                curr_drift_Y = np.abs(ops.nodeDisp(top_nodes[i], 2) - ops.nodeDisp(bottom_nodes[i], 2))/h[i]

                # Check if the current drift is greater than the previous peaks at the same storey
                if curr_drift_X > peak_drift[i, 0]:
                    peak_drift[i, 0] = curr_drift_X

                if curr_drift_Y > peak_drift[i, 1]:
                    peak_drift[i, 1] = curr_drift_Y

            # For each node to monitor, get is absolute displacement
            for i in np.arange(len(control_nodes)):

                curr_disp_X = np.abs(ops.nodeDisp(control_nodes[i], 1))
                curr_disp_Y = np.abs(ops.nodeDisp(control_nodes[i], 2))

                # # Append the node displacements and accelerations (NOTE: Might change when bidirectional records are applied)
                # node_disps[step,i]  = ops.nodeDisp(control_nodes[i],1)
                # node_accels[step,i] = ops.nodeResponse(control_nodes[i],1, 3)

                # Check if the current drift is greater than the previous peaks at the same storey
                if curr_disp_X > peak_disp[i, 0]:
                    peak_disp[i, 0] = curr_disp_X

                if curr_disp_Y > peak_disp[i, 1]:
                    peak_disp[i, 1] = curr_disp_Y

        # Now that the analysis is finished, get the maximum in either direction and report the location also
        max_peak_drift = np.max(peak_drift)
        ind = np.where(peak_drift == max_peak_drift)
        if ind[1][0] == 0:
            max_peak_drift_dir = 'X'
        elif ind[1][0] == 1:
            max_peak_drift_dir = 'Y'
        max_peak_drift_loc = ind[0][0]+1

        # Get the floor accelerations. Need to use a recorder file because a direct query would return relative values
        ops.wipe() # First wipe to finish writing to the file

        if len(fnames) > 0:
            temp1 = np.transpose(np.max(np.abs(np.loadtxt(f"{nrha_outdir}/floor_accel_X.txt")), 0))
            peak_accel[:,0] = temp1
            os.remove(f"{nrha_outdir}/floor_accel_X.txt")

        elif len(fnames) > 1:

            temp1 = np.transpose(np.max(np.abs(np.loadtxt(f"{nrha_outdir}/floor_accel_X.txt")), 0))
            temp2 = np.transpose(np.max(np.abs(np.loadtxt(f"{nrha_outdir}/floor_accel_Y.txt")), 0))
            peak_accel = np.stack([temp1, temp2], axis=1)
            os.remove(f"{nrha_outdir}/floor_accel_X.txt")
            os.remove(f"{nrha_outdir}/floor_accel_Y.txt")

        # Get the maximum in either direction and report the location also
        max_peak_accel = np.max(peak_accel)
        ind = np.where(peak_accel == max_peak_accel)
        if ind[1][0] == 0:
            max_peak_accel_dir = 'X'
        elif ind[1][0] == 1:
            max_peak_accel_dir = 'Y'
        max_peak_accel_loc = ind[0][0]

        # Give some feedback on what happened
        if conv_index == -1:
            print('------ ANALYSIS FAILED --------')
        elif conv_index == 0:
            print('~~~~~~~ ANALYSIS SUCCESSFUL ~~~~~~~~~')

        if pflag is True:
            print('Final state = {:d} (-1 for non-converged, 0 for stable)'.format(conv_index))
            print('Maximum peak storey drift {:.3f} radians at storey {:d} in the {:s} direction (Storeys = 1, 2, 3,...)'.format(max_peak_drift, max_peak_drift_loc, max_peak_drift_dir))
            print('Maximum peak floor acceleration {:.3f} g at floor {:d} in the {:s} direction (Floors = 0(G), 1, 2, 3,...)'.format(max_peak_accel, max_peak_accel_loc, max_peak_accel_dir))

        # Give the outputs
        return control_nodes, conv_index, peak_drift, peak_accel, max_peak_drift, max_peak_drift_dir, max_peak_drift_loc, max_peak_accel, max_peak_accel_dir, max_peak_accel_loc, peak_disp
