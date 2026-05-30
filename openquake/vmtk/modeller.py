import os
import numpy as np
import matplotlib.pyplot as plt
import platform
if platform.system() == "Windows":
    import openseespy.opensees as ops
elif platform.system() == "Darwin":
    import openseespymac.opensees as ops
else:
    import openseespy.opensees as ops
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from openquake.vmtk.units import units
from openquake.vmtk.plotter import plotter


class modeller:
    """
    Model and analyse multi-degree-of-freedom (MDOF) oscillators
    using OpenSees.

    This class provides functionality to create, analyse, and
    visualise structural models for dynamic and static analyses,
    including gravity analysis, modal analysis, static pushover
    analysis (SPO), cyclic pushover analysis (CPO), nonlinear
    time-history analysis (NRHA), incremental dynamic analysis
    (IDA), and earthquake-sequence NRHA.

    The model uses a stick-and-mass idealisation with zero-length
    Pinching4 hysteretic springs wrapped in a MinMax material for
    automatic collapse detection. Floor accelerations in all NRHA
    methods are recorded as absolute (total) accelerations (relative
    acceleration plus ground-motion input), which is physically
    correct for assessing acceleration-sensitive non-structural
    components.

    Attributes
    ----------
    number_storeys : int
        The number of storeys in the building model.
    storey_heights : list
        List of storey heights in metres.
    floor_masses : list
        List of floor masses in tonnes.
    storey_drifts : numpy.ndarray
        Array of inter-storey displacement capacities, shape
        ``(number_storeys, cap_points)`` where *cap_points* is 2
        (bilinear), 3 (trilinear), or 4 (quadrilinear).
    storey_forces : numpy.ndarray
        Array of storey shear-force capacities with the same shape
        as *storey_drifts*.
    degradation : bool
        If ``True``, Pinching4 hysteretic degradation is enabled.

    Methods
    -------
    create_Pinching4_material(mat1Tag, mat2Tag, storey_forces,
            storey_drifts, degradation)
        Creates a Pinching4 + MinMax material pair for one storey.

    compile_model()
        Builds the MDOF stick model (nodes, masses, boundary
        conditions, zero-length elements) in OpenSees.

    plot_model(pFlag=True, export_path=None)
        Plots a 2-D visualisation of the stick-and-mass model
        (node elevations with spring symbols and backbone curves).

    do_gravity_analysis(...)
        Performs gravity analysis on the MDOF system.

    do_modal_analysis(...)
        Performs modal analysis to determine natural frequencies
        and mode shapes.

    do_spo_analysis(ref_disp, disp_scale_factor, push_dir, phi, ...)
        Performs static pushover analysis (SPO) on the MDOF system.

    do_cpo_analysis(ref_disp, mu_levels, push_dir, dispIncr, phi,
            ..., max_step=None)
        Performs cyclic pushover analysis (CPO) on the MDOF system
        with MinMax stopping criteria.

    do_nrha_analysis(fnames, dt_gm, sf, t_max, dt_ansys, ...)
        Performs nonlinear time-history analysis (NRHA) on the MDOF
        system with absolute-acceleration recording and MinMax
        stopping criteria.

    do_ida_analysis(fnames, dt_gm, t_max, dt_ansys,
            ...)
        Performs incremental dynamic analysis (IDA) using the
        Hunt-Trace-Fill algorithm (Vamvatsikos & Cornell, 2002).

    do_nrha_analysis_sequences(fnames, time_vector, sf, ...)
        Performs NRHA for concatenated earthquake sequences with
        variable time-steps, automatic record-boundary detection,
        per-sequence peak-drift and hysteretic-energy reporting,
        absolute-acceleration recording, and MinMax stopping
        criteria. The time vector is aligned to the OpenSees
        ``-dt`` / ``-filePath`` convention by prepending ``t = 0``
        when the supplied vector starts at ``dt > 0``.

    """

    def __init__(
        self,
        number_storeys,
        storey_heights,
        floor_masses,
        storey_drifts,
        storey_forces,
        degradation,
    ):
        """
        Initializes the modeller object and validates the input parameters.

        Parameters
        ----------
        number_storeys : int
            The number of storeys in the building model.

        storey_heights : list
            List of storey heights in meters (e.g., [2.5, 3.0]).

        floor_masses : list
            List of floor masses in tonnes (e.g., [1000, 1200]).

        storey_drifts : np.array
            Array of inter-storey displacements
            (size = number of storeys, CapPoints).

        storey_forces : np.array
            Array of storey forces (size = number of storeys, CapPoints).

        degradation : bool
            Boolean to enable or disable hysteresis degradation.

        Raises
        ------
        TypeError
            If any input has an incorrect type.

        ValueError
            If any input has an invalid value or inconsistent dimensions.
        """

        # number_storeys check
        if not isinstance(number_storeys, int) or number_storeys < 1:
            raise ValueError("'number_storeys' must be a positive integer.")

        # storey_heights check
        if not hasattr(storey_heights, '__len__'):
            raise TypeError("'storey_heights' must be a list or array.")
        if len(storey_heights) != number_storeys:
            raise ValueError(
                f"'storey_heights' length ({len(storey_heights)}) "
                f"must match 'number_storeys' ({number_storeys})."
            )
        if any(h <= 0 for h in storey_heights):
            raise ValueError(
                "All values in 'storey_heights' must be positive.")

        # floor_masses check
        if not hasattr(floor_masses, '__len__'):
            raise TypeError("'floor_masses' must be a list or array.")
        if len(floor_masses) != number_storeys:
            raise ValueError(
                f"'floor_masses' length ({len(floor_masses)}) "
                f"must match 'number_storeys' ({number_storeys})."
            )
        if any(m <= 0 for m in floor_masses):
            raise ValueError("All values in 'floor_masses' must be positive.")

        # storey_drifts and storey_forces check
        storey_drifts = np.atleast_2d(storey_drifts)
        storey_forces = np.atleast_2d(storey_forces)

        if storey_drifts.shape[0] != number_storeys:
            raise ValueError(
                f"'storey_drifts' must have {number_storeys} rows "
                f"(one per storey), got {storey_drifts.shape[0]}."
            )
        if storey_forces.shape[0] != number_storeys:
            raise ValueError(
                f"'storey_forces' must have {number_storeys} rows "
                f"(one per storey), got {storey_forces.shape[0]}."
            )
        if storey_drifts.shape != storey_forces.shape:
            raise ValueError(
                f"'storey_drifts' and 'storey_forces' must have "
                f"the same shape, got {storey_drifts.shape} "
                f"and {storey_forces.shape}."
            )
        cap_points = storey_drifts.shape[1]
        if cap_points not in (2, 3, 4):
            raise ValueError(
                f"Each storey must have 2, 3, or 4 capacity points "
                f"(bilinear/trilinear/quadrilinear), "
                f"got {cap_points}."
            )
        if np.any(storey_drifts <= 0):
            raise ValueError("All values in 'storey_drifts' must be positive.")
        if np.any(storey_forces <= 0):
            raise ValueError("All values in 'storey_forces' must be positive.")
        for i in range(number_storeys):
            if not np.all(np.diff(storey_drifts[i]) > 0):
                raise ValueError(
                    f"'storey_drifts' for storey {i + 1} must be "
                    f"strictly increasing.")

        # degradation check
        if not isinstance(degradation, bool):
            raise TypeError(
                f"'degradation' must be a bool, "
                f"got {type(degradation).__name__}.")

        self.number_storeys = number_storeys
        self.storey_heights = storey_heights
        self.floor_masses = floor_masses
        self.storey_drifts = storey_drifts
        self.storey_forces = storey_forces
        self.degradation = degradation

    def create_Pinching4_material(
        self,
        mat1Tag,
        mat2Tag,
        storey_forces,
        storey_drifts,
        degradation,
    ):
        """
        Creates a Pinching4 material model for the multi-degree-of-freedom
        material object in stick model analysis.

        The Pinching4 material model is used to simulate hysteretic behavior
        in structures under dynamic loading,
        including degradation if enabled. The method assigns the material
        properties to the building storeys based
        on the given parameters.

        Parameters
        ----------
        mat1Tag : int
            Material tag for the first material in the Pinching4 model.

        mat2Tag : int
            Material tag for the second material in the Pinching4 model.

        storey_forces : np.array
            Array of storey forces at each storey in the model.

        storey_drifts : np.array
            Array of storey displacements corresponding to the forces.

        degradation : bool
            Boolean flag to enable or disable hysteresis degradation in the
            Pinching4 material model.

        Returns
        -------
        None
            This method does not return any value but modifies the internal
            material definitions for the model.

        References:
        -----------
        1) Vamvatsikos D (2011) Software—earthquake, steel dynamics and
           probability, viewed January 2021.
           http://users.ntua.gr/divamva/software.html

        2) Martins, L., Silva, V., Crowley, H. et al. Vulnerability
           modellers toolkit, an open-source platform for vulnerability
           analysis. Bull Earthquake Eng 19, 5691-5709 (2021).
           https://doi.org/10.1007/s10518-021-01187-w

        3) Minjie Zhu, Frank McKenna, Michael H. Scott, OpenSeesPy: Python
           library for the OpenSees finite element framework, SoftwareX,
           Volume 7, 2018, Pages 6-11, ISSN 2352-7110,
           https://doi.org/10.1016/j.softx.2017.10.009.
           (https://www.sciencedirect.com/science/article/
           pii/S2352711017300584)

        Notes
        -----
        The `mat1Tag` and `mat2Tag` represent different materials used in
        the Pinching4 hysteretic model, where the degradation flag controls
        the material's degradation behavior during the simulation.
        """

        force = np.zeros([5, 1])
        disp = np.zeros([5, 1])

        # Bilinear
        if len(storey_forces) == 2:

            # Force values for bilinear curve are assigned based on the
            # first and last points of the storey capacity curve
            force[1] = storey_forces[0]
            force[4] = storey_forces[-1]
            # Displacement values for bilinear curve are assigned based on
            # the first and last points of the storey capacity curve
            disp[1] = storey_drifts[0]
            disp[4] = storey_drifts[-1]
            # Intermediate disp points: divide range into 3 equal parts
            disp[2] = disp[1] + (disp[4] - disp[1]) / 3
            disp[3] = disp[1] + 2 * ((disp[4] - disp[1]) / 3)
            # Interpolate forces at intermediate displacement points
            force[2] = np.interp(disp[2], storey_drifts, storey_forces)
            force[3] = np.interp(disp[3], storey_drifts, storey_forces)

        # Trilinear
        elif len(storey_forces) == 3:

            # Force values: first, last, and second point of curve
            force[1] = storey_forces[0]
            force[4] = storey_forces[-1]
            # Displacement values: first and last points of curve
            disp[1] = storey_drifts[0]
            disp[4] = storey_drifts[-1]
            # First intermediate point: second capacity curve point
            force[2] = storey_forces[1]
            disp[2] = storey_drifts[1]
            # Second intermediate: midpoint between pt2 and end
            disp[3] = np.mean([disp[2], disp[-1]])
            force[3] = np.interp(disp[3], storey_drifts, storey_forces)

        # Quadrilinear
        elif len(storey_forces) == 4:

            # Force values: first and last points of capacity curve
            force[1] = storey_forces[0]
            force[4] = storey_forces[-1]
            # Displacement values: first and last points of curve
            disp[1] = storey_drifts[0]
            disp[4] = storey_drifts[-1]
            # First intermediate point: second capacity curve point
            force[2] = storey_forces[1]
            disp[2] = storey_drifts[1]
            # Second intermediate point: third capacity curve point
            force[3] = storey_forces[2]
            disp[3] = storey_drifts[2]

        if degradation is True:
            matargs = [
                force[1, 0],
                disp[1, 0],
                force[2, 0],
                disp[2, 0],
                force[3, 0],
                disp[3, 0],
                force[4, 0],
                disp[4, 0],
                -1 * force[1, 0],
                -1 * disp[1, 0],
                -1 * force[2, 0],
                -1 * disp[2, 0],
                -1 * force[3, 0],
                -1 * disp[3, 0],
                -1 * force[4, 0],
                -1 * disp[4, 0],
                0.5,
                0.25,
                0.05,
                0.5,
                0.25,
                0.05,
                0,
                0.1,
                0,
                0,
                0.2,
                0,
                0.1,
                0,
                0,
                0.2,
                0,
                0.4,
                0,
                0.4,
                0.9,
                10,
                "energy",
            ]
        else:
            matargs = [
                force[1, 0],
                disp[1, 0],
                force[2, 0],
                disp[2, 0],
                force[3, 0],
                disp[3, 0],
                force[4, 0],
                disp[4, 0],
                -1 * force[1, 0],
                -1 * disp[1, 0],
                -1 * force[2, 0],
                -1 * disp[2, 0],
                -1 * force[3, 0],
                -1 * disp[3, 0],
                -1 * force[4, 0],
                -1 * disp[4, 0],
                0.5,
                0.25,
                0.05,
                0.5,
                0.25,
                0.05,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                10,
                "energy",
            ]

        # Create the Pinching4 material in OpenSees with the defined parameters
        ops.uniaxialMaterial("Pinching4", mat1Tag, *matargs)
        # Create a MinMax material in OpenSees to define the limits of the
        # hysteretic behavior based on the maximum positive and negative
        # displacements, ensuring that the material response is constrained
        # within these bounds during the analysis
        ops.uniaxialMaterial(
            "MinMax", mat2Tag, mat1Tag, "-min", -1 * disp[-1, 0], "-max",
            disp[-1, 0]
        )

    def compile_model(self):
        """
        Compiles and sets up the multi-degree-of-freedom (MDOF) oscillator
        model in OpenSees.

        This method constructs the model by defining nodes, assigning masses,
        imposing boundary conditions, and creating elements with associated
        material models for each storey in the building structure.
        It also defines rigid elastic materials for restrained degrees of
        freedom and nonlinear materials
        for unrestrained degrees of freedom. The method finally assembles
        the model for dynamic analysis.

        The process involves:

        1. Initializing the OpenSees model.
        2. Creating base and floor nodes.
        3. Assigning masses and degrees of freedom.
        4. Applying boundary conditions for the nodes.
        5. Creating zero-length elements for each storey with their
           respective material properties.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        - The method uses OpenSees' `ops.node`, `ops.mass`, and `ops.element`
          to define nodes, masses, and zero-length elements for the MDOF
          oscillator.
        - Boundary conditions are applied with the base node being fully
          fixed, while the upper storeys have horizontal degrees of freedom
          released.
        - The material model used for each storey is a Pinching4 hysteretic
          model, created by the `create_Pinching4_material` method.
        """

        # Set model builder
        ops.wipe()  # wipe existing model
        ops.model("basic", "-ndm", 3, "-ndf", 6)

        # Define base node (tag = 0)
        ops.node(0, *[0.0, 0.0, 0.0])

        # Define floor nodes (tag = 1+)
        current_height = 0.0

        # Use range based on the length of heights to ensure we never
        # go out of bounds
        for i in range(len(self.storey_heights)):

            # Node tags start from 1 for the first floor node
            nodeTag = i + 1
            # Accumulate storey height to get current node elevation
            current_height += self.storey_heights[i]
            # Assign the corresponding floor mass for the current node
            current_mass = self.floor_masses[i]
            # Define node coordinates (X, Y, Z) with Z being the current height
            coords = [0.0, 0.0, current_height]
            # Assign mass to X and Y translations
            masses = [current_mass, current_mass, 1e-9, 1e-9, 1e-9, 1e-9]
            # Create the node and assign mass
            ops.node(nodeTag, *coords)
            ops.mass(nodeTag, *masses)

        # Update number_storeys to match the actual number of nodes created
        self.number_storeys = len(self.storey_heights)

        # Get list of model nodes
        nodeList = ops.getNodeTags()
        # Impose boundary conditions
        for i in nodeList:
            # fix the base node against all DOFs
            if i == 0:
                ops.fix(i, 1, 1, 1, 1, 1, 1)
            # release the horizontal DOFs (1,2) and fix remaining
            else:
                ops.fix(i, 0, 0, 1, 1, 1, 1)

        # Get number of zerolength elements required
        nodeList = ops.getNodeTags()

        for i in range(self.number_storeys):

            # Define the material tag associated with each storey
            mat1Tag = int(f"1{i}00")  # hysteretic material tag
            mat2Tag = int(f"1{i}01")  # min-max material tag

            # Extract backbone curve (drifts and forces) for this storey
            current_storey_drifts = self.storey_drifts[
                i, :
            ].tolist()
            current_storey_forces = self.storey_forces[
                i, :
            ].tolist()

            # Create rigid elastic materials for the restrained dofs
            rigM = int(f"1{i}02")
            ops.uniaxialMaterial("Elastic", rigM, 1e16)

            # Create the nonlinear material for the unrestrained dofs
            self.create_Pinching4_material(
                mat1Tag,
                mat2Tag,
                current_storey_forces,
                current_storey_drifts,
                self.degradation,
            )

            # Define element connectivity
            eleTag = int(f"200{i}")
            eleNodes = [i, i + 1]

            # Create the element
            ops.element(
                "zeroLength",
                eleTag,
                eleNodes[0],
                eleNodes[1],
                "-mat",
                mat2Tag,
                mat2Tag,
                rigM,
                rigM,
                rigM,
                rigM,
                "-dir",
                1,
                2,
                3,
                4,
                5,
                6,
                "-doRayleigh",
                1,
            )

    def plot_model(self, pFlag=True, export_path=None):
        """
        Plots a 2-D visualisation of the stick-and-mass model as two subplots.

        Left panel  — node elevation diagram with zigzag spring symbols
        (zero-length Pinching4 elements), individual storey height brackets,
        and a separate double-headed total-height dimension line.
        Right panel — force-deformation backbone curves for every storey,
        both axes starting from zero.
        Legend boxes are placed at the same vertical level below each panel.

        Parameters
        ----------
        pFlag : bool, optional, default=True
            If True, the plot is processed (saved/shown).

        export_path : str, optional
            Full path including filename to save the plot.
            Creates directories if missing.

        Returns
        -------
        None
        """

        # Get list of model nodes to extract their coordinates and masses
        # for plotting
        nodeList = ops.getNodeTags()
        NodeCoordListZ, NodeMassList = [], []
        for tag in nodeList:
            NodeCoordListZ.append(ops.nodeCoord(tag, 3))
            NodeMassList.append(ops.nodeMass(tag, 1))

        n_st = self.number_storeys
        total_h = max(NodeCoordListZ)

        # Define colors for different plot elements
        COL_BASE = "#B71C1C"
        COL_NODE = "#1565C0"
        COL_ANN = "#37474F"
        COL_GRID = "#EBEBEB"
        COL_SPRING = "#546E7A"
        BG = "white"
        s_colors = [plt.cm.tab10(i % 10) for i in range(n_st)]

        # Custom function to draw zigzag spring symbols representing the
        # zero-length Pinching4 elements
        def _draw_spring(ax, x, z_bot, z_top, color, n_teeth=6, width=0.06):
            pad = (z_top - z_bot) * 0.15
            n_pts = n_teeth * 2 + 1
            zs = np.linspace(z_bot + pad, z_top - pad, n_pts)
            xs = np.empty(n_pts)
            xs[0] = x
            xs[-1] = x
            for k in range(1, n_pts - 1):
                xs[k] = x + width if k % 2 == 1 else x - width
            ax.plot([x, x], [z_bot, z_bot + pad],
                    color=color, lw=1.5, zorder=3)
            ax.plot([x, x], [z_top - pad, z_top],
                    color=color, lw=1.5, zorder=3)
            ax.plot(
                xs,
                zs,
                color=color,
                lw=1.5,
                zorder=3,
                solid_capstyle="round",
                solid_joinstyle="round",
            )

        # Custom legend handler that draws a zigzag spring icon for the
        # zero-length Pinching4 elements in the legend box, ensuring that
        # the legend accurately represents the spring symbols used in the
        # node elevation diagram, enhancing the clarity and visual appeal of
        # the plot while maintaining consistency with the overall design and
        # color scheme
        class _SpringHandler:
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                x0, y0 = handlebox.xdescent, handlebox.ydescent
                w, h = handlebox.width, handlebox.height
                n = 4
                n_pts = n * 2 + 1
                xs_l = np.linspace(x0 + 2, x0 + w - 2, n_pts)
                ys_l = np.empty(n_pts)
                cy = y0 + h / 2
                amp = h * 0.38
                ys_l[0] = cy
                ys_l[-1] = cy
                for k in range(1, n_pts - 1):
                    ys_l[k] = cy + amp if k % 2 == 1 else cy - amp
                line = mlines.Line2D(
                    xs_l,
                    ys_l,
                    color=COL_SPRING,
                    lw=1.5,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                )
                handlebox.add_artist(line)
                return line

        # Create the figure and axes for the two subplots, setting the
        # background color and defining the layout with specified width
        # ratios to ensure that the node elevation diagram and
        # force-deformation backbones are visually balanced and clearly
        # distinguishable, while maintaining a cohesive color scheme
        # throughout the plot
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(10, 6), gridspec_kw={"width_ratios": [1, 2.2]}
        )
        fig.patch.set_facecolor(BG)
        ax1.set_facecolor(BG)
        ax2.set_facecolor(BG)

        col_x = 0.0

        for i in range(n_st):
            _draw_spring(
                ax1, col_x,
                NodeCoordListZ[i], NodeCoordListZ[i + 1],
                COL_SPRING
            )

        for i, (z, m) in enumerate(zip(NodeCoordListZ, NodeMassList)):
            mk = "s" if i == 0 else "o"
            co = COL_BASE if i == 0 else COL_NODE
            sz = 160 if i == 0 else 120
            ax1.scatter(
                col_x,
                z,
                s=sz,
                marker=mk,
                color=co,
                edgecolors="white",
                linewidths=1.2,
                zorder=5,
            )
            ax1.plot(
                [col_x + 0.02, col_x + 0.09], [z, z],
                lw=0.7, color="#B0BEC5", zorder=1
            )
            ax1.text(
                col_x + 0.11,
                z,
                f"Node {i}   z = {z:.2f} m   m = {m:.3f} t",
                fontsize=7,
                color=COL_ANN,
                va="center",
                ha="left",
                fontfamily="monospace",
            )

        # Storey brackets (left of spring)
        bx_st = -0.12  # vertical bar of individual storey brackets
        for i in range(n_st):
            z_bot = NodeCoordListZ[i]
            z_top = NodeCoordListZ[i + 1]
            z_mid = (z_bot + z_top) / 2.0
            sh = z_top - z_bot
            ax1.plot([bx_st - 0.03, bx_st], [z_bot, z_bot],
                     lw=0.7, color="#90A4AE")
            ax1.plot([bx_st - 0.03, bx_st], [z_top, z_top],
                     lw=0.7, color="#90A4AE")
            ax1.plot(
                [bx_st - 0.03, bx_st - 0.03], [z_bot, z_top],
                lw=0.7, color="#90A4AE"
            )
            ax1.text(
                bx_st - 0.05,
                z_mid,
                f"{sh:.2f} m",
                fontsize=6,
                color="#90A4AE",
                ha="right",
                va="center",
            )

        # Element ID labels (right of spring)
        ele_list = ops.getEleTags()
        for i in range(n_st):
            z_mid = (NodeCoordListZ[i] + NodeCoordListZ[i + 1]) / 2.0
            ele_id = ele_list[i] if i < len(ele_list) else i
            ax1.text(
                col_x + 0.09,
                z_mid,
                f"Ele. {ele_id}",
                fontsize=6.5,
                color=COL_SPRING,
                ha="left",
                va="center",
                style="italic",
            )

        # Axis styling
        for sp in ["top", "right", "bottom"]:
            ax1.spines[sp].set_visible(False)
        ax1.spines["left"].set_color("#90A4AE")
        ax1.spines["left"].set_linewidth(0.8)
        ax1.set_xticks([])
        ax1.set_xlim(-0.40, 1.45)
        ax1.set_ylim(0, total_h + 0.5)
        ax1.set_ylabel(
            "Height,  z  [m]",
            fontsize=9, fontweight="bold", color=COL_ANN, labelpad=6
        )
        ax1.tick_params(axis="y", labelsize=8, colors=COL_ANN)
        ax1.set_title(
            "Node Positions",
            fontsize=10, fontweight="bold", color="#1A237E", pad=8
        )

        ax2.grid(True, color=COL_GRID, linewidth=0.7, zorder=0)
        ax2.set_axisbelow(True)

        for i in range(n_st):
            d = np.concatenate(([0.0], self.storey_drifts[i, :]))
            f = np.concatenate(([0.0], self.storey_forces[i, :]))
            sc = s_colors[i]
            ax2.plot(
                d,
                f,
                color=sc,
                lw=1.8,
                zorder=3,
                label=f"Storey {i + 1}",
                solid_capstyle="round",
            )
            ax2.scatter(
                d[1:],
                f[1:],
                color=sc,
                s=25,
                zorder=4,
                edgecolors="white",
                linewidths=0.6,
            )

        for sp in ["top", "right"]:
            ax2.spines[sp].set_visible(False)
        ax2.spines["left"].set_color(COL_ANN)
        ax2.spines["left"].set_linewidth(1.0)
        ax2.spines["bottom"].set_color(COL_ANN)
        ax2.spines["bottom"].set_linewidth(1.0)

        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0)
        ax2.set_xlabel(
            "Storey Drift Capacity,  \u03b4\u1d62  [m]",
            fontsize=9,
            fontweight="bold",
            color=COL_ANN,
            labelpad=7,
        )
        ax2.set_ylabel(
            "Storey Shear Force,  V\u1d62  [kN]",
            fontsize=9,
            fontweight="bold",
            color=COL_ANN,
            labelpad=7,
        )
        ax2.tick_params(labelsize=8, colors=COL_ANN)
        ax2.set_title(
            "Storey Force\u2013Deformation Relationships",
            fontsize=10,
            fontweight="bold",
            color="#1A237E",
            pad=8,
        )

        # Legends: same vertical level below each panel
        spring_handle = Line2D(
            [], [], color=COL_SPRING, lw=1.5, label="Zero-length spring"
        )
        handles1 = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=COL_BASE,
                markersize=7,
                label="Fixed base node",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COL_NODE,
                markersize=7,
                label="Floor node",
            ),
            spring_handle,
        ]
        handles2 = [
            Line2D([0], [0], color=s_colors[i],
                   lw=1.8, label=f"Storey {i + 1}")
            for i in range(n_st)
        ]

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.16)
        fig.canvas.draw()

        p1 = ax1.get_position()
        p2 = ax2.get_position()
        leg_y = 0.01

        fig.legend(
            handles=handles1,
            handler_map={spring_handle: _SpringHandler()},
            fontsize=7.5,
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(p1.x0 + p1.width / 2, leg_y),
            bbox_transform=fig.transFigure,
            framealpha=0.95,
            edgecolor="#CFD8DC",
            borderpad=0.6,
            handletextpad=0.4,
        )
        fig.legend(
            handles=handles2,
            fontsize=7.5,
            ncol=min(n_st, 5),
            loc="lower center",
            bbox_to_anchor=(p2.x0 + p2.width / 2, leg_y),
            bbox_transform=fig.transFigure,
            framealpha=0.95,
            edgecolor="#CFD8DC",
            borderpad=0.6,
            handletextpad=0.4,
        )

        # Super-title
        label = "SDOF Oscillator" if n_st == 1 else f"{n_st}-Storey MDOF"
        fig.suptitle(
            f"OpenSees {label}  \u2014  Stick-and-Mass Model",
            fontsize=11,
            fontweight="bold",
            color="#1A237E",
            y=1.01,
        )

        # Save or Show
        if pFlag:
            if export_path:
                directory = os.path.dirname(export_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                plt.savefig(export_path, dpi=300)
                plt.show()
            else:
                plt.show()
        else:
            plt.close()

    ##########################################################################
    #                             ANALYSIS MODULES                           #
    ##########################################################################
    def do_gravity_analysis(
        self,
        nG=100,
        ansys_soe="UmfPack",
        constraints_handler="Transformation",
        numberer="RCM",
        test_type="NormDispIncr",
        init_tol=1.0e-6,
        init_iter=500,
        algorithm_type="Newton",
        integrator="LoadControl",
        analysis="Static",
    ):
        """
        Perform a gravity analysis on a multi-degree-of-freedom
        (MDOF) system in OpenSees.

        This method sets up and runs a gravity analysis using specified
        parameters for various analysis objects in OpenSees. The gravity
        analysis solves for the static equilibrium of the system under
        self-weight loads (e.g., gravity loads).

        Parameters
        ----------
        nG: int, optional
            Number of gravity analysis steps to perform. Default is 100.

        ansys_soe: string, optional
            The system of equations type to be used in the analysis.
            This defines how the system of equations will be solved.
            Default is 'UmfPack' (sparse direct solver).

        constraints_handler: string, optional
            The constraints handler determines how the constraint
            equations are enforced in the analysis. It controls the
            enforcement of specified values for degrees-of-freedom
            (DOFs) or relationships between them. Default is
            'Transformation' (transforming the constrained DOFs into
            active ones).

        numberer: string, optional
            The degree-of-freedom numberer defines how DOFs are
            numbered. This is important for system efficiency in
            solving. Default is 'RCM' (Reverse Cuthill-McKee, a
            reordering algorithm).

        test_type: string, optional
            Defines the test type used to check the convergence of the
            solution. It is used in constructing the LinearSOE and
            LinearSolver objects. Default is 'NormDispIncr' (norm of
            displacement increment).

        init_tol: float, optional
            The tolerance criterion for checking convergence. A smaller
            value means stricter convergence. Default is 1.0e-6.

        init_iter: int, optional
            The maximum number of iterations to check for convergence.
            Default is 500.

        algorithm_type: string, optional
            Defines the solution algorithm used in the analysis. Common
            options are 'Newton' (Newton-Raphson) for solving the
            system of equations. Default is 'Newton'.

        integrator: string, optional
            Defines the integrator for the analysis. The integrator
            dictates how the analysis steps are taken in time or load.
            Default is 'LoadControl' (control load increments).

        analysis: string, optional
            Defines the type of analysis to be performed. 'Static' is
            typically used for gravity analysis, but other options
            (e.g., 'Transient') can be used depending on the type of
            analysis. Default is 'Static'.

        Returns
        -------
        None.

        Notes
        -----
        - This method sets up the analysis using OpenSees by defining the
          system of equations, constraints handler, numberer, convergence
          test, solution algorithm, integrator, and analysis type.
        - The gravity analysis solves for the static equilibrium under
          self-weight or gravity loads and is typically used to determine
          the initial equilibrium state of a structure before dynamic loading.
        - The analysis can be modified by changing the parameters to adjust
          solver settings, tolerance, and other relevant options.
        - After the analysis is completed, the analysis objects are wiped to
          ensure a clean state for further analyses.
        """

        # Define the analysis objects and run gravity analysis
        # System of equations: sparse solver with partial pivoting
        ops.system(ansys_soe)
        # Constraints handler: enforces DOF constraints
        ops.constraints(constraints_handler)
        # DOF numberer: controls equation numbering order
        ops.numberer(numberer)
        # Convergence test: checks solution convergence
        ops.test(test_type, init_tol, init_iter, 3)
        # Solution algorithm: iterative solver (e.g., Newton-Raphson)
        ops.algorithm(algorithm_type)
        # Integrator: controls load increment steps
        ops.integrator(integrator, (1 / nG))
        # Analysis type: static gravity analysis
        ops.analysis(analysis)
        # Run analysis for nG steps to reach gravity equilibrium
        ops.analyze(nG)
        # Reset time to zero for subsequent analyses
        ops.loadConst("-time", 0.0)
        # Cleanup analysis objects for a clean subsequent state
        ops.wipeAnalysis()

    def do_modal_analysis(
        self,
        num_modes=3,
        solver="-genBandArpack",
        doRayleigh=False,
        pFlag=False,
        plot_modes=True,
        export_path=None,
    ):
        """
        Perform modal analysis on a multi-degree-of-freedom (MDOF)
        system to determine its natural frequencies and mode shapes.

        This method calculates the natural frequencies and corresponding
        mode shapes of the system. The natural frequencies are
        determined by solving the eigenvalue problem, and the mode
        shapes are normalized for the system's degrees of freedom. The
        results can be used to assess the dynamic characteristics of
        the system.

        Parameters
        ----------
        num_modes: int, optional
            The number of modes to consider in the analysis. Default
            is 3. This parameter determines how many modes will be
            computed in the modal analysis.

        solver: string, optional
            The type of solver to use for the eigenvalue problem.
            Default is '-genBandArpack', which uses a generalized
            banded Arnoldi method for large sparse eigenvalue problems.

        doRayleigh: bool, optional
            Flag to enable or disable Rayleigh damping in the modal
            analysis. This parameter is not used directly in this
            method but can be set in the OpenSees model. Default is
            False.

        pFlag: bool, optional
            Flag to control whether to print the modal analysis
            report. If True, the fundamental period and mode shape
            will be printed to the console. Default is False.

        plot_modes: bool, optional
            Flag to control whether to plot the modes. If True, the
            mode shapes are plotted against the undeformed shape.
            Default is True

        export_path: str, optional
            If a string path is provided (e.g., 'modal_results.png'),
            the plot will be saved to this location. If None, the plot
            will be only displayed and not saved. Default is None.

        Returns
        -------
        T: array
            The periods of vibration for the system, calculated as
            2π/ω, where ω are the natural frequencies obtained from
            the eigenvalue problem.

        mode_shape: list
            A list of the normalized mode shapes for the system, with
            each element representing the displacement in the
            x-direction for the corresponding mode. The mode shapes
            are normalized by the last node's displacement.
        """

        # Solve eigenvalue problem and compute natural frequencies and T
        self.omega = np.power(ops.eigen(solver, num_modes), 0.5)
        T = 2.0 * np.pi / self.omega

        # Get node list for mode shape extraction
        node_list = ops.getNodeTags()

        # Extract and normalise mode shapes for each mode
        mode_shape_vectors = []
        for mode_num in range(1, num_modes + 1):
            # Extract X, Y, Z displacements for all nodes
            ux_all = np.array(
                [ops.nodeEigenvector(tag, mode_num, 1)
                 for tag in node_list]
            )
            uy_all = np.array(
                [ops.nodeEigenvector(tag, mode_num, 2)
                 for tag in node_list]
            )
            uz_all = np.array(
                [ops.nodeEigenvector(tag, mode_num, 3)
                 for tag in node_list]
            )

            # Stack into (n_nodes x 3) mode shape array
            mode_vector = np.column_stack((ux_all, uy_all, uz_all))

            # Normalization
            max_disp = np.max(np.abs(mode_vector))
            if max_disp != 0:
                mode_vector /= max_disp
            mode_shape_vectors.append(mode_vector)

        # Print modal analysis report if pFlag is True
        if pFlag:
            ops.modalProperties("-print")
            print(f"Fundamental Period: T = {T[0]:.3f} s")

        # Plot the mode shapes if plot_modes is True
        if plot_modes:
            # Initialise the plotter class
            pl = plotter()
            pl.plot_modes(node_list, mode_shape_vectors,
                          T, export_path=export_path)

        # Internal cleanup of analysis objects
        ops.wipeAnalysis()

        return T, mode_shape_vectors

    def do_spo_analysis(
        self,
        ref_disp,
        disp_scale_factor,
        push_dir,
        phi,
        pFlag=True,
        num_steps=200,
        ansys_soe="BandGeneral",
        constraints_handler="Transformation",
        numberer="RCM",
        test_type="EnergyIncr",
        init_tol=1.0e-5,
        init_iter=1000,
        algorithm_type="KrylovNewton",
        save_animation_path=None,
    ):
        """
        Perform static pushover analysis (SPO) on a stick model.
        This method simulates a static pushover analysis where a lateral
        load pattern is incrementally applied to the structure. The
        displacement at the control node is increased step by step, and the
        corresponding base shear, floor displacements, and forces in
        non-linear elements are recorded. The analysis helps in evaluating
        the structural response to lateral loads, such as earthquake forces.
        evaluating the structural response to lateral loads.

        Parameters
        ----------
        ref_disp: float
            The reference displacement at which the analysis starts,
            corresponding to the yield or other significant
            displacement (e.g., 1mm).

        disp_scale_factor: float
            The scale factor applied to the reference displacement to
            determine the final displacement. The analysis will be run
            to this scaled displacement.

        push_dir: int
            The direction in which the pushover load is applied:
                1 = X direction
                2 = Y direction
                3 = Z direction

        phi: list of floats
            The lateral load pattern shape. This is typically a mode
            shape or a predefined load distribution. For example, it
            can be the first-mode shape from the calibrate_model
            function.

        pFlag: bool, optional
            Flag to print (or not) the pushover analysis steps. If
            True, detailed feedback on each step will be printed.
            Default is True.

        num_steps: int, optional
            The number of steps to increment the pushover load.
            Default is 200.

        ansys_soe: string, optional
            The type of system of equations solver to use. Default is
            'BandGeneral'.

        constraints_handler: string, optional
            The constraints handler object to determine how constraint
            equations are enforced. Default is 'Transformation'.

        numberer: string, optional
            The degree-of-freedom (DOF) numberer object to determine
            the mapping between equation numbers and
            degrees-of-freedom. Default is 'RCM'.

        test_type: string, optional
            The type of test to use for the linear system of
            equations. Default is 'EnergyIncr'.

        init_tol: float, optional
            The tolerance criterion to check for convergence.
            Default is 1.0e-5.

        init_iter: int, optional
            The maximum number of iterations to perform when checking
            for convergence. Default is 1000.

        algorithm_type: string, optional
            The type of algorithm used to solve the system. Default
            is 'KrylovNewton'.

        save_animation_path: string, optional,
            If provided, saves the figure to this path
            (e.g., 'spo.gif')

        Returns
        -------
        spo_dict : dict
            A dictionary containing the SPO results with the following keys:

            - ``'spo_disps'``: array, shape (TimeSteps × Floors) — floor displacements.
            - ``'spo_rxn'``: array, shape (TimeSteps,) — base shear at each step.
            - ``'spo_disps_spring'``: array, shape (TimeSteps × Springs) — spring displacements.
            - ``'spo_forces_spring'``: array, shape (TimeSteps × Springs) — spring shear forces.
            - ``'spo_idr'``: array, shape (TimeSteps × Storeys) — interstorey drift ratio history.
            - ``'spo_midr'``: array, shape (TimeSteps,) — maximum IDR across all storeys.
        """

        # Set up linear time series and plain load pattern
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)

        # Identify control node, pattern nodes, and reaction nodes
        nodeList = ops.getNodeTags()
        # Control node: top node monitored for displacement
        control_node = nodeList[-1]
        # Pattern nodes: floor nodes receiving lateral loads
        pattern_nodes = nodeList[1:]
        # Reaction nodes: base node(s) for base shear recording
        rxn_nodes = [nodeList[0]]

        # Apply lateral loads to floor nodes scaled by phi and mass
        for i in np.arange(len(pattern_nodes)):
            load_val = 1.0 if len(
                pattern_nodes) == 1 else phi[i] * self.floor_masses[i]
            if push_dir == 1:
                ops.load(
                    pattern_nodes[i], load_val, 0.0, 0.0, 0.0, 0.0, 0.0)
            elif push_dir == 2:
                ops.load(
                    pattern_nodes[i], 0.0, load_val, 0.0, 0.0, 0.0, 0.0)
            elif push_dir == 3:
                ops.load(
                    pattern_nodes[i], 0.0, 0.0, load_val, 0.0, 0.0, 0.0)

        # Set up analysis objects
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)
        ops.algorithm(algorithm_type)

        # Displacement-control integrator toward target displacement
        target_disp = float(ref_disp) * float(disp_scale_factor)
        delta_disp = target_disp / (1.0 * num_steps)
        ops.integrator("DisplacementControl",
                       control_node, push_dir, delta_disp)
        ops.analysis("Static")

        # Get element list for spring force/displacement recording
        elementList = ops.getEleTags()

        # Print analysis header if requested
        if pFlag is True:
            print(
                f"\n------ Static Pushover Analysis of Node "
                f"# {control_node} to {target_disp} ---------"
            )

        # Initialize convergence flag, step counter, and load factor
        ok = 0
        step = 1
        loadf = 1.0

        # Initialize result arrays for base shear, displacements,
        # spring deformations, and spring forces
        spo_rxn = np.array([0.0])
        spo_top_disp = np.array(
            [ops.nodeResponse(control_node, push_dir, 1)]
        )  # Used for animation and Pushover Curve
        spo_disps = np.array(
            [[ops.nodeResponse(node, push_dir, 1) for node in pattern_nodes]]
        )
        spo_disps_spring = np.array(
            [[ops.eleResponse(ele, "deformation")[0] for ele in elementList]]
        )
        spo_forces_spring = np.array(
            [[ops.eleResponse(ele, "force")[0] for ele in elementList]]
        )

        # Main pushover loop: step to target displacement
        while step <= num_steps and ok == 0 and loadf > 0:

            # Perform one analysis step and check convergence
            ok = ops.analyze(1)

            # Adaptive convergence: relax criteria if step fails
            if ok != 0:
                if pFlag:
                    print("FAILED! Trying relaxing convergence.")
                ops.test(test_type, init_tol * 0.01, init_iter)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                if pFlag:
                    print(
                        "FAILED! Trying relaxing convergence "
                        "with more iterations.")
                ops.test(test_type, init_tol * 0.01, init_iter * 10)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                if pFlag:
                    print(
                        "FAILED! Trying relaxing convergence with "
                        "more iterations and Newton with "
                        "InitialThenCurrent."
                    )
                ops.test(test_type, init_tol * 0.01, init_iter * 10)
                ops.algorithm("Newton", "initialThenCurrent")
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                if pFlag:
                    print(
                        "FAILED! Trying relaxing convergence with "
                        "more iterations and Newton with initial."
                    )
                ops.test(test_type, init_tol * 0.01, init_iter * 10)
                ops.algorithm("Newton", "initial")
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                if pFlag:
                    print("FAILED! Attempting a Hail Mary.")
                ops.test("FixedNumIter", init_iter * 10)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
                # Final check before breaking
                if ok != 0:
                    break

            # Get current load factor (time)
            loadf = ops.getTime()

            # Print step progress if requested
            if pFlag is True:
                curr_disp = ops.nodeDisp(control_node, push_dir)
                print(
                    f"Currently pushed node {control_node} to "
                    f"{curr_disp:.4f} with load factor {loadf:.4f}"
                )

            # Advance step counter
            step += 1

            # Record results: base shear, displacements, spring forces
            spo_top_disp = np.append(
                spo_top_disp, ops.nodeResponse(control_node, push_dir, 1)
            )

            current_disps = np.array(
                [ops.nodeResponse(node, push_dir, 1) for node in pattern_nodes]
            )
            spo_disps = np.append(spo_disps, np.array([current_disps]), axis=0)

            spo_disps_spring = np.append(
                spo_disps_spring,
                np.array(
                    [[ops.eleResponse(ele, "deformation")[0]
                      for ele in elementList]]
                ),
                axis=0,
            )

            spo_forces_spring = np.append(
                spo_forces_spring,
                np.array([[ops.eleResponse(ele, "force")[0]
                         for ele in elementList]]),
                axis=0,
            )

            ops.reactions()
            temp = 0
            for n in rxn_nodes:
                temp += ops.nodeReaction(n, push_dir)
            spo_rxn = np.append(spo_rxn, -temp)

        # Check final convergence status and print results if pFlag is True
        if ok != 0:
            print("------ ANALYSIS FAILED! --------")
        elif ok == 0:
            print("~~~~~~~ ANALYSIS SUCCESSFUL! ~~~~~~~~~")
        if loadf < 0:
            print("Stopped because of load factor below zero")

        # Cleanup analysis objects
        ops.wipeAnalysis()

        # Calculate Interstorey Drift Ratio (IDR) from floor disps
        idr_disps = spo_disps.copy()

        # Prepend ground floor (zero displacement)
        ground_disps = np.zeros((idr_disps.shape[0], 1))
        full_idr_disps = np.hstack([ground_disps, idr_disps])

        # Compute interstorey displacements (ISD)
        spo_isd = np.diff(full_idr_disps, axis=1)

        # Convert storey_heights to a numpy array for division
        storey_heights = np.array(self.storey_heights)

        # Normalize by corresponding storey heights to get IDR (x100 requested)
        spo_idr = (spo_isd / storey_heights) * 100

        # Take the maximum interstorey drift ratio per step
        spo_midr = np.max(np.abs(spo_idr), axis=1)

        # Handle Animation (Call updated function with spo_midr)
        if save_animation_path:
            pl = plotter()
            pl.animate_spo(
                spo_top_disp,
                spo_rxn,
                spo_disps,
                spo_midr,
                nodeList,
                elementList,
                push_dir,
                phi,
                save_animation_path,
            )

        # Pack and Return results into a dictionary
        spo_dict = {
            "spo_disps": spo_disps,
            "spo_rxn": spo_rxn,
            "spo_disps_spring": spo_disps_spring,
            "spo_forces_spring": spo_forces_spring,
            "spo_idr": spo_idr,
            "spo_midr": spo_midr,
        }

        return spo_dict

    def do_cpo_analysis(
        self,
        ref_disp,
        mu_levels,
        push_dir,
        dispIncr,
        phi,
        pFlag=True,
        max_step=None,
        ansys_soe="BandGeneral",
        constraints_handler="Transformation",
        numberer="RCM",
        test_type="NormDispIncr",
        init_tol=1.0e-5,
        init_iter=1000,
        algorithm_type="KrylovNewton",
        save_animation_path=None,
    ):
        """
        Perform cyclic pushover analysis (CPO) on a stick model.
        This method simulates a cyclic pushover analysis where a lateral
        load pattern is incrementally applied to the structure. This
        procedure simulates a cyclic lateral loading protocol in which
        the stick model is subjected to alternating displacement-controlled
        loading cycles at the control node. This approach allows the
        evaluation of strength degradation, stiffness deterioration, pinching
        effects, and energy dissipation capacity under repeated lateral
        loading, providing a more realistic representation of structural
        behaviour under seismic cyclic demands compared to a monotonic
        pushover analysis.

        Parameters
        ----------
        ref_disp: float
            Reference displacement (e.g., yield displacement) for scaling
            the cycles.

        mu_levels: list
            Target ductility factors (mu) for each cycle level.

        push_dir: int
            Direction of the pushover analysis (1=X, 2=Y, 3=Z).

        dispIncr: int
            Minimum number of displacement increments per half-cycle. Used
            directly when ``max_step`` is None, or as a lower bound when
            ``max_step`` is set. A value of 5-10 is typical.

        max_step: float or None, optional, default=None
            Maximum displacement increment size per analysis step [m]. When
            set, ``numIncr`` for each half-cycle is computed as
            ``max(dispIncr, ceil(excursion / max_step))``, so larger cycles
            are automatically subdivided more finely and all loops stay smooth.
            For example, setting ``max_step = ref_disp * 0.1`` limits each
            step to 10% of the reference displacement. Set to ``None`` to
            use a fixed ``dispIncr`` for every half-cycle.

        phi: list of floats
            The lateral load pattern shape vector (scaled by mass).

        pFlag: bool, optional, default=True
            If True, prints feedback during the analysis steps.

        save_animation_path: str, optional, default=None
            If provided, the path to save the animation
            (e.g., 'cpo.gif' or 'cpo.mp4').

        ansys_soe: string, optional, default='BandGeneral'
            System of equations solver.

        constraints_handler: string, optional, default='Transformation'
            Constraint handler method.

        numberer: string, optional, default='RCM'
            The numberer method.

        test_type: string, optional, default='NormDispIncr'
            Convergence test type.

        init_tol: float, optional, default=1.0e-5
            The initial tolerance for convergence.

        init_iter: int, optional, default=1000
            The maximum number of iterations for the solver.

        algorithm_type: string, optional, default='KrylovNewton
            The solution algorithm type.

        save_animation_path: string, optional,
            If provided, saves the figure to this path (e.g., 'cpo.gif')

        Returns
        -------
        cpo_dict: dict
            A dictionary containing all the analysis results.
        """

        # Define the analysis objects
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)

        # Get all tags needed for analysis and animation
        nodeList = ops.getNodeTags()
        elementList = ops.getEleTags()

        # Control node: top node, pattern nodes: floor nodes
        control_node = nodeList[-1]
        pattern_nodes = nodeList[1:]
        # Reaction nodes: base node(s) for base shear recording
        rxn_nodes = [nodeList[0]]

        # Quality control
        assert len(phi) == len(
            pattern_nodes), "phi length must match pattern_nodes"
        assert len(self.floor_masses) == len(
            pattern_nodes
        ), "floor_masses length mismatch"

        # Apply lateral load pattern scaled by mass
        for i in np.arange(len(pattern_nodes)):
            if push_dir == 1:
                ops.load(
                    pattern_nodes[i],
                    phi[i] * self.floor_masses[i],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            elif push_dir == 2:
                ops.load(
                    pattern_nodes[i],
                    0.0,
                    phi[i] * self.floor_masses[i],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            elif push_dir == 3:
                ops.load(
                    pattern_nodes[i],
                    0.0,
                    0.0,
                    phi[i] * self.floor_masses[i],
                    0.0,
                    0.0,
                    0.0,
                )

        # Set up analysis objects
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)
        ops.algorithm(algorithm_type)

        # Build target displacement list: +mu, -mu, +2mu, -2mu, ...
        cycleDispList = []
        for mu in mu_levels:
            cycleDispList.append(ref_disp * mu)  # push positive
            cycleDispList.append(-ref_disp * mu)  # pull negative
        dispNoMax = len(cycleDispList)

        if pFlag:
            print(
                f"\n------ Cyclic Pushover with ductility "
                f"levels: {mu_levels} ------")

        # MinMax deformation limits — same approach as do_nrha_analysis.
        # The MinMax material kills the spring once deformation exceeds the
        # ultimate displacement (last column of storey_drifts).  We use the
        # same 1.5x limit so CPO and NRHA collapse detection are consistent.
        minmax_limits = 1.0 * np.abs(self.storey_drifts[:, -1])  # (n_storeys,)
        elementList_cpo = ops.getEleTags()

        # Recording data arrays
        cpo_rxn = [0.0]
        cpo_top_disp = [ops.nodeDisp(control_node, push_dir)]
        cpo_disps = [[ops.nodeDisp(node, push_dir) for node in pattern_nodes]]
        energy_steps = [0.0]
        minmax_failed = False  # flag: MinMax spring limit exceeded

        for d in range(dispNoMax):
            if minmax_failed:
                break

            current_disp = ops.nodeDisp(control_node, push_dir)
            target_disp = cycleDispList[d]
            excursion = abs(target_disp - current_disp)

            # Adaptive step count: if max_step is given, use enough increments
            # so no single step exceeds max_step. Always honour dispIncr as a
            # minimum so short early cycles are not over-subdivided.
            if max_step is not None and max_step > 0:
                numIncr = max(dispIncr, int(np.ceil(excursion / max_step)))
            else:
                numIncr = dispIncr

            dU = (target_disp - current_disp) / numIncr

            # Use DisplacementControl integrator
            ops.integrator("DisplacementControl", control_node, push_dir, dU)
            ops.analysis("Static")

            # Loop over displacement increments
            for incr in range(numIncr):
                ok = ops.analyze(1)

                # Convergence Failure Handling (Extended Recovery)
                if ok != 0:
                    print(
                        f"FAILED at cycle {d + 1}/{dispNoMax}, "
                        f"increment {incr}/{numIncr}: "
                        f"Starting recovery attempts..."
                    )

                # Try relaxing convergence tolerance
                if ok != 0:
                    print("FAILED: Trying relaxing convergence...")
                    ops.test(test_type, init_tol * 0.01, init_iter)
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)

                # Try relaxing convergence tolerance with more iterations
                if ok != 0:
                    print(
                        "FAILED: Trying relaxing convergence "
                        "with more iterations...")
                    ops.test(test_type, init_tol * 0.01, init_iter * 10)
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)

                # Try relaxing tolerance with Newton 'initialThenCurrent'
                if ok != 0:
                    print(
                        "FAILED: Trying relaxing convergence with "
                        "more iteration and Newton with "
                        "initial then current..."
                    )
                    ops.test(test_type, init_tol * 0.01, init_iter * 10)
                    ops.algorithm("Newton", "initialThenCurrent")
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)
                    ops.algorithm(algorithm_type)

                # Try relaxing tolerance with Newton 'initial'
                if ok != 0:
                    print(
                        "FAILED: Trying relaxing convergence with "
                        "more iteration and Newton with initial..."
                    )
                    ops.test(test_type, init_tol * 0.01, init_iter * 10)
                    ops.algorithm("Newton", "initial")
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)
                    ops.algorithm(algorithm_type)  # Restore original algorithm

                # Attempt a Hail Mary (FixedNumIter)
                if ok != 0:
                    print("FAILED: Attempting a Hail Mary...")
                    ops.test("FixedNumIter", init_iter * 10)
                    ok = ops.analyze(1)
                    ops.test(
                        test_type, init_tol, init_iter
                    )  # Restore original test type

                # Final failure check
                if ok != 0:
                    print("Analysis Failed")
                    break

                # Data Recording (only if successful)
                if ok == 0:
                    curr_disp = ops.nodeDisp(control_node, push_dir)
                    cpo_top_disp.append(curr_disp)

                    current_floor_disps = [
                        ops.nodeDisp(node, push_dir) for node in pattern_nodes
                    ]
                    cpo_disps.append(current_floor_disps)

                    ops.reactions()
                    temp = sum(ops.nodeReaction(n, push_dir)
                               for n in rxn_nodes)
                    curr_rxn = -temp
                    cpo_rxn.append(curr_rxn)

                    if len(cpo_top_disp) >= 2:
                        dU_step = cpo_top_disp[-1] - cpo_top_disp[-2]
                        avg_F = 0.5 * (cpo_rxn[-1] + cpo_rxn[-2])
                        dE = abs(avg_F * dU_step)
                        energy_steps.append(energy_steps[-1] + dE)
                    else:
                        energy_steps.append(energy_steps[-1])

                    # Check if any MinMax material limit has been
                    # exceeded (spring deformation >= 1.5 * ultimate
                    # storey drift) — stop analysis if so.
                    for s_idx, ele in enumerate(elementList_cpo):
                        try:
                            deform_result = ops.eleResponse(
                                ele, "deformation")
                            if deform_result is None:
                                if pFlag:
                                    print(
                                        f"MinMax material killed "
                                        f"spring at storey "
                                        f"{s_idx + 1} "
                                        f"— stopping CPO analysis."
                                    )
                                minmax_failed = True
                                break
                            spring_deform = abs(deform_result[0])
                            if spring_deform >= minmax_limits[s_idx]:
                                if pFlag:
                                    print(
                                        f"MinMax material failed at "
                                        f"storey {s_idx + 1} "
                                        f"(deform="
                                        f"{spring_deform:.4f} >= "
                                        f"limit="
                                        f"{minmax_limits[s_idx]:.4f}"
                                        f") — stopping CPO analysis."
                                    )
                                minmax_failed = True
                                break
                        except Exception:
                            if pFlag:
                                print(
                                    f"MinMax material killed spring "
                                    f"at storey {s_idx + 1} "
                                    f"(eleResponse failed) "
                                    f"— stopping CPO analysis."
                                )
                            minmax_failed = True
                            break

                    if minmax_failed:
                        break

            if minmax_failed:
                break

            if pFlag is True:
                curr_disp = ops.nodeDisp(control_node, push_dir)
                print(
                    f"Cycle target {d + 1}/{dispNoMax}: Pushed node "
                    f"{control_node} to {curr_disp:.4f}"
                )

        # Convert lists to numpy arrays
        cpo_rxn = np.array(cpo_rxn)
        cpo_top_disp = np.array(cpo_top_disp)
        cpo_disps = np.array(cpo_disps)
        pseudo_steps = np.arange(len(energy_steps))
        cpo_energy = np.column_stack((pseudo_steps, energy_steps))

        # Calculate Interstorey Drifts
        base_disps = np.zeros((cpo_disps.shape[0], 1))
        padded_disps = np.hstack((base_disps, cpo_disps))
        cpo_drifts = np.diff(padded_disps, axis=1)
        max_interstorey_drift = np.max(np.abs(cpo_drifts))

        ops.wipeAnalysis()

        # Final output dictionary (cpo_dict)
        cpo_dict = {
            "cpo_disps": cpo_disps,
            "cpo_rxn": cpo_rxn,
            "cpo_top_disp": cpo_top_disp,
            "cpo_energy": cpo_energy,
            "cpo_idr": cpo_drifts,
            "cpo_midr": max_interstorey_drift,
        }

        # Animation Call
        if save_animation_path:
            pl = plotter()
            pl.animate_cpo(
                cpo_dict, nodeList, elementList, push_dir, save_animation_path
            )

        return cpo_dict

    def do_nrha_analysis(
        self, fnames, dt_gm, sf, t_max, dt_ansys,
        pFlag=True, xi=0.05, ansys_soe='BandGeneral',
        constraints_handler='Plain', numberer='RCM',
        test_type='NormDispIncr', init_tol=1.0e-6, init_iter=50,
        algorithm_type='Newton', save_animation_path=None,
        drift_thresholds=None
    ):
        """
        Perform nonlinear time-history analysis on a
        Multi-Degree-of-Freedom (MDOF) system.

        Supports uni-directional (1 file) and bi-directional (2 files)
        ground motion loading. Floor accelerations are recorded as
        absolute (total) accelerations, including at the base, which
        is physically correct for assessing acceleration-sensitive
        non-structural components. Hysteretic energy dissipation is
        computed via signed force-velocity integration (trapezoidal
        rule), correctly capturing only dissipated energy and not
        elastic recovery.

        Parameters
        ----------
        fnames: list
            List of file paths to the ground motion records. One file
            applies X-direction loading; two files apply
            bi-directional (X and Y) loading simultaneously.

        dt_gm: float
            Time-step of the ground motion records.

        sf: float
            Scale factor to apply to the ground motion records.
            Typically equal to the gravitational acceleration
            (9.81 m/s²) when records are in units of g.

        t_max: float
            The maximum time duration for the analysis.

        dt_ansys: float
            The integration time-step for the analysis. Typically
            smaller than dt_gm.

        pFlag: bool, optional, default=True
            If True, prints progress updates during the analysis.

        xi: float, optional, default=0.05
            Inherent viscous damping ratio (default is 5%).

        ansys_soe: string, optional, default='BandGeneral'
            System of equations solver type.

        constraints_handler: string, optional, default='Plain'
            Method used to enforce constraint equations.

        numberer: string, optional, default='RCM'
            DOF numberer object (Reverse Cuthill-McKee by default).

        test_type: string, optional, default='NormDispIncr'
            Convergence test type.

        init_tol: float, optional, default=1.0e-6
            Convergence tolerance.

        init_iter: int, optional, default=50
            Maximum number of iterations per time step.

        algorithm_type: string, optional, default='Newton'
            Nonlinear solution algorithm.

        save_animation_path: string, optional
            If provided, saves the NRHA animation to this file path
            (e.g., 'nrha.gif').

        drift_thresholds: list, optional
            Drift thresholds used in the animation for damage-state
            colour changes.

        Returns
        -------
        control_nodes: list
            List of all node tags in the model (base node first,
            then floor nodes).

        conv_index: int
            Convergence status: 0 = success, -1 = failure.

        peak_drift: numpy.ndarray
            Peak inter-storey drift ratio per storey, shape
            (n_storeys, 2). Column 0 = X, column 1 = Y direction.

        peak_accel: numpy.ndarray
            Peak absolute floor acceleration (in g) per node (base
            + floors), shape (n_nodes, 2). Column 0 = X, column 1 =
            Y direction. Row 0 is the base (ground motion PGA).

        max_peak_drift: float
            Maximum peak inter-storey drift ratio across all storeys
            and directions.

        max_peak_drift_dir: string
            Direction ('X' or 'Y') of the maximum peak drift.

        max_peak_drift_loc: int
            Storey number (1-based) of the maximum peak drift.

        max_peak_accel: float
            Maximum peak absolute floor acceleration (g) across all
            floors and directions.

        max_peak_accel_dir: string
            Direction ('X' or 'Y') of the maximum peak acceleration.

        max_peak_accel_loc: int
            Floor number (0-based, 0 = base/ground) of the maximum
            peak acceleration.

        peak_disp: numpy.ndarray
            Peak relative displacement (m) per node, shape
            (n_nodes, 2). Column 0 = X, column 1 = Y.

        hysteretic_energy_per_storey: numpy.ndarray
            Dissipated hysteretic energy per storey (kN·m), shape
            (n_storeys,). Computed via signed F·v trapezoidal
            integration to capture only true energy dissipation
            (not elastic strain energy recovery).

        total_hysteretic_energy: float
            Total dissipated hysteretic energy summed across all
            storeys (kN·m).
        """

        # MinMax deformation limits (1.0 * ultimate storey disp) per element
        # storey_drifts shape: (number_storeys, CapPoints); last col = ult.
        # one limit per storey
        minmax_limits = 1.0 * np.abs(self.storey_drifts[:, -1])

        #  Determine loading directions from fnames
        bidir = len(fnames) >= 2   # True -> apply both X and Y ground motions

        # Define control nodes
        control_nodes = ops.getNodeTags()
        n_nodes = len(control_nodes)

        #  Apply ground motion time-series and uniform excitation patterns
        if len(fnames) > 0:
            ops.timeSeries('Path', 1, '-dt', dt_gm, '-filePath',
                           fnames[0], '-factor', sf)
            ops.pattern('UniformExcitation', 1, 1, '-accel', 1)   # X direction
        if len(fnames) > 1:
            ops.timeSeries('Path', 2, '-dt', dt_gm, '-filePath',
                           fnames[1], '-factor', sf)
            ops.pattern('UniformExcitation', 2, 2, '-accel', 2)   # Y direction
        if len(fnames) > 2:
            ops.timeSeries('Path', 3, '-dt', dt_gm, '-filePath',
                           fnames[2], '-factor', sf)
            # Z direction (rarely used)
            ops.pattern('UniformExcitation', 3, 3, '-accel', 3)

        #  Configure analysis objects
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)
        ops.algorithm(algorithm_type)
        ops.integrator('Newmark', 0.5, 0.25)
        ops.analysis('Transient')

        # Set up analysis parameters
        conv_index = 0     # -1 = failure, 0 = success
        # time [s] at which collapse/non-convergence first detected
        collapse_time = None
        control_time = 0.0
        ok = 0

        #  Build storey height array for IDR normalisation
        if n_nodes < 2:
            top_nodes = []
            bottom_nodes = []
        else:
            top_nodes = control_nodes[1:]
            bottom_nodes = control_nodes[:-1]

        h = []
        for i in range(len(top_nodes)):
            topZ = ops.nodeCoord(top_nodes[i], 3)
            bottomZ = ops.nodeCoord(bottom_nodes[i], 3)
            dist = topZ - bottomZ
            if dist == 0:
                print(
                    "WARNING: Zero storey height detected "
                    "— using 1e9 to avoid division by zero.")
                h.append(1e9)
            else:
                h.append(dist)
        h = np.array(h) if len(h) > 0 else np.array([])

        # Pre-allocate recording arrays
        n_steps = int(np.ceil(t_max / dt_ansys)) + 1

        # Relative displacements (X and Y separately)
        node_disps_x = np.zeros((n_steps, n_nodes))
        node_disps_y = np.zeros((n_steps, n_nodes))

        # Absolute (total) accelerations in g (X and Y separately)
        node_accels_x = np.zeros((n_steps, n_nodes))
        node_accels_y = np.zeros((n_steps, n_nodes))

        # Peak trackers — shape (n_nodes, 2): col 0 = X, col 1 = Y
        peak_disp = np.zeros((n_nodes, 2))
        peak_accel = np.zeros((n_nodes, 2))

        # Peak IDR tracker — shape (n_storeys, 2): col 0 = X, col 1 = Y
        peak_drift = np.zeros((len(top_nodes), 2))

        # Include Rayleigh damping
        num_frequencies = len(self.omega)

        if num_frequencies == 1:
            # SDOF case
            alphaM = 2 * self.omega[0] * xi
            ops.rayleigh(alphaM, 0, 0, 0)
        elif num_frequencies >= 2:
            # MDOF case: use first and last mode (up to 3rd)
            idx_high = min(num_frequencies - 1, 2)
            alphaM = 2 * self.omega[0] * self.omega[idx_high] * \
                xi / (self.omega[0] + self.omega[idx_high])
            alphaK = 2 * xi / (self.omega[0] + self.omega[idx_high])
            ops.rayleigh(alphaM, 0, alphaK, 0)

        # Hysteretic energy tracking
        # Uses signed F·v integration (trapezoidal) so only genuine
        # dissipation is accumulated; combined X+Y resultant for bidir.
        element_tags = ops.getEleTags()
        n_elements = len(element_tags)

        # Previous-step values for trapezoidal integration
        # For bidir, track both components: [X, Y] per element
        energy_force_prev_x = np.zeros(n_elements)
        energy_force_prev_y = np.zeros(n_elements)
        energy_vel_prev_x = np.zeros(n_elements)
        energy_vel_prev_y = np.zeros(n_elements)
        hysteretic_energy_per_storey = np.zeros(n_elements)
        energy_time_prev = 0.0

        # Progress print throttle
        print_every = max(1, int(np.ceil(n_steps / 50.0)))

        #  Main time-stepping loop
        step = 0
        while conv_index == 0 and control_time <= t_max and ok == 0:

            ok = ops.analyze(1, dt_ansys)
            control_time = ops.getTime()

            if pFlag and (step % print_every == 0 or control_time >= t_max):
                print(f'Completed {control_time:.3f} of {t_max:.3f} seconds')

            # Adaptive convergence recovery
            if ok != 0:
                print(
                    'FAILED at {:.3f}: Trying half '
                    'time-step.'.format(control_time))
                ok = ops.analyze(1, 0.5 * dt_ansys)
            if ok != 0:
                print(
                    'FAILED at {:.3f}: Trying quarter '
                    'time-step.'.format(control_time))
                ok = ops.analyze(1, 0.25 * dt_ansys)
            if ok != 0:
                print(
                    'FAILED at {:.3f}: Relaxing convergence '
                    '+ more iterations.'.format(control_time))
                ops.test(test_type, init_tol * 0.01, init_iter * 10)
                ok = ops.analyze(1, 0.5 * dt_ansys)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                print('FAILED at {:.3f}: Newton initialThenCurrent.'.format(
                    control_time))
                ops.test(test_type, init_tol * 0.01, init_iter * 10)
                ops.algorithm('Newton', 'initialThenCurrent')
                ok = ops.analyze(1, 0.5 * dt_ansys)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                print('FAILED at {:.3f}: Newton initial.'.format(
                    control_time))
                ops.test(test_type, init_tol * 0.01, init_iter * 10)
                ops.algorithm('Newton', 'initial')
                ok = ops.analyze(1, 0.5 * dt_ansys)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                print('FAILED at {:.3f}: Hail Mary (FixedNumIter).'.format(
                    control_time))
                ops.test('FixedNumIter', init_iter * 10)
                ok = ops.analyze(1, 0.5 * dt_ansys)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                print('FAILED at {:.3f}: Exiting analysis.'.format(
                    control_time))
                conv_index = -1
                collapse_time = control_time
                break

            # Grow arrays if step counter reached allocated size
            if step >= node_disps_x.shape[0]:
                extra = max(100, node_disps_x.shape[0])
                node_disps_x = np.concatenate(
                    [node_disps_x,
                     np.zeros((extra, n_nodes))], axis=0)
                node_disps_y = np.concatenate(
                    [node_disps_y,
                     np.zeros((extra, n_nodes))], axis=0)
                node_accels_x = np.concatenate(
                    [node_accels_x,
                     np.zeros((extra, n_nodes))], axis=0)
                node_accels_y = np.concatenate(
                    [node_accels_y,
                     np.zeros((extra, n_nodes))], axis=0)

            # Record nodal responses

            for i, node in enumerate(control_nodes):

                # Relative displacements (used for IDR and peak_disp)
                disp_x = ops.nodeDisp(node, 1)
                disp_y = ops.nodeDisp(node, 2)
                node_disps_x[step, i] = disp_x
                node_disps_y[step, i] = disp_y

                # Peak relative displacement
                if abs(disp_x) > peak_disp[i, 0]:
                    peak_disp[i, 0] = abs(disp_x)
                if abs(disp_y) > peak_disp[i, 1]:
                    peak_disp[i, 1] = abs(disp_y)

                # Absolute acceleration = relative accel + ground accel.
                # ops.nodeAccel() returns RELATIVE acceleration.
                # ops.getLoadFactor() returns ground accel in m/s²
                # (already scaled by sf); sum gives absolute value.
                ag_x = ops.getLoadFactor(1) if len(fnames) > 0 else 0.0
                ag_y = ops.getLoadFactor(2) if len(fnames) > 1 else 0.0
                abs_accel_x = (ops.nodeAccel(node, 1) + ag_x) / \
                    9.81   # convert m/s² → g
                abs_accel_y = (ops.nodeAccel(node, 2) + ag_y) / 9.81
                node_accels_x[step, i] = abs_accel_x
                node_accels_y[step, i] = abs_accel_y

                # Peak absolute acceleration
                if abs(abs_accel_x) > peak_accel[i, 0]:
                    peak_accel[i, 0] = abs(abs_accel_x)
                if abs(abs_accel_y) > peak_accel[i, 1]:
                    peak_accel[i, 1] = abs(abs_accel_y)

            #  Inter-storey drift ratios
            if len(top_nodes) > 0:
                # X direction
                dx_top = node_disps_x[step, 1:]
                dx_bottom = node_disps_x[step, :-1]
                idr_x = np.abs(dx_top - dx_bottom) / h
                mask_x = idr_x > peak_drift[:, 0]
                peak_drift[mask_x, 0] = idr_x[mask_x]

                # Y direction (only meaningful if bidir loading applied)
                dy_top = node_disps_y[step, 1:]
                dy_bottom = node_disps_y[step, :-1]
                idr_y = np.abs(dy_top - dy_bottom) / h
                mask_y = idr_y > peak_drift[:, 1]
                peak_drift[mask_y, 1] = idr_y[mask_y]

            # Hysteretic energy — signed F·v trapezoidal integration
            # Energy = ∫ F · v_interstory dt
            # Positive when force and velocity are in the same direction
            # (loading); negative during elastic unloading.
            # Cumulative sum thus represents only dissipated energy.
            # For bidir: combine X and Y contributions per storey.
            energy_time_curr = control_time
            dt_energy = energy_time_curr - energy_time_prev
            if dt_energy > 0:
                for ei, ele_tag in enumerate(element_tags):
                    ele_force_vec = ops.eleForce(ele_tag)
                    force_curr_x = ele_force_vec[0]   # X-direction shear
                    force_curr_y = ele_force_vec[1]   # Y-direction shear

                    node_top = control_nodes[ei + 1]
                    node_bot = control_nodes[ei]

                    # Inter-storey velocity components
                    vel_curr_x = ops.nodeVel(
                        node_top, 1) - ops.nodeVel(node_bot, 1)
                    vel_curr_y = ops.nodeVel(
                        node_top, 2) - ops.nodeVel(node_bot, 2)

                    # Signed power at current and previous step (trapezoidal)
                    power_prev = (
                        energy_force_prev_x[ei] * energy_vel_prev_x[ei]
                        + energy_force_prev_y[ei] * energy_vel_prev_y[ei]
                    )
                    power_curr = (force_curr_x * vel_curr_x +
                                  force_curr_y * vel_curr_y)
                    dE = 0.5 * (power_prev + power_curr) * dt_energy
                    hysteretic_energy_per_storey[ei] += dE

                    # Update previous-step values
                    energy_force_prev_x[ei] = force_curr_x
                    energy_force_prev_y[ei] = force_curr_y
                    energy_vel_prev_x[ei] = vel_curr_x
                    energy_vel_prev_y[ei] = vel_curr_y

            energy_time_prev = energy_time_curr
            step += 1

            # MinMax spring failure check
            # If eleResponse returns None or raises, the spring has
            # been killed by OpenSees — treat as collapse immediately.
            for s_idx, ele in enumerate(element_tags):
                try:
                    deform_result = ops.eleResponse(ele, 'deformation')
                    if deform_result is None:
                        if pFlag:
                            print(
                                f'COLLAPSE DETECTED: Spring at '
                                f'storey {s_idx + 1} killed by MinMax'
                                f' at t={control_time:.3f}s.')
                        conv_index = -1
                        collapse_time = control_time
                        break
                    spring_deform = abs(deform_result[0])
                    if spring_deform >= minmax_limits[s_idx]:
                        if pFlag:
                            print(
                                f'COLLAPSE DETECTED! Spring at '
                                f'storey {s_idx + 1} reached MinMax '
                                f'limit ({spring_deform:.4f} >= '
                                f'{minmax_limits[s_idx]:.4f}) '
                                f'at t={control_time:.3f}s. '
                                f'Capping drift and terminating.')
                        conv_index = -1
                        collapse_time = control_time
                        break
                except Exception:
                    if pFlag:
                        print(
                            f'COLLAPSE DETECTED! Spring at storey '
                            f'{s_idx + 1} unresponsive (MinMax killed)'
                            f' at t={control_time:.3f}s.')
                    conv_index = -1
                    collapse_time = control_time
                    break
            if conv_index == -1:
                break

        #  Trim arrays to actual number of completed steps
        node_disps_x = node_disps_x[:step, :]
        node_disps_y = node_disps_y[:step, :]
        node_accels_x = node_accels_x[:step, :]
        node_accels_y = node_accels_y[:step, :]

        # Keep node_disps (X) and node_accels (X) for animation
        node_disps = node_disps_x.copy()
        node_accels = node_accels_x.copy()

        #  Maximum drift summary
        max_peak_drift = np.max(peak_drift) if peak_drift.size > 0 else 0.0
        if peak_drift.size > 0:
            ind = np.unravel_index(np.argmax(peak_drift), peak_drift.shape)
            max_peak_drift_dir = 'X' if ind[1] == 0 else 'Y'
            max_peak_drift_loc = ind[0] + 1   # 1-based storey number
        else:
            max_peak_drift_dir = 'X'
            max_peak_drift_loc = 0

        # Maximum acceleration summary
        # peak_accel already updated per-step using nodeAccel (absolute).
        max_peak_accel = np.max(peak_accel)
        if peak_accel.size > 0:
            ind_a = np.unravel_index(np.argmax(peak_accel), peak_accel.shape)
            max_peak_accel_dir = 'X' if ind_a[1] == 0 else 'Y'
            max_peak_accel_loc = ind_a[0]     # 0-based floor (0 = base)
        else:
            max_peak_accel_dir = 'X'
            max_peak_accel_loc = 0

        #  Total hysteretic energy
        total_hysteretic_energy = float(np.sum(hysteretic_energy_per_storey))

        #  Console feedback
        if conv_index == -1:
            print('------ ANALYSIS FAILED --------')
        else:
            print('~~~~~~~ ANALYSIS SUCCESSFUL ~~~~~~~~~')

        if pFlag:
            direction_label = (
                'bi-directional (X+Y)' if bidir
                else 'uni-directional (X)')
            print(f'Loading: {direction_label}')
            print(
                'Final state = {:d} (-1 for non-converged, '
                '0 for stable)'.format(conv_index))
            print(
                'Maximum peak storey drift {:.4f} at storey {:d} '
                'in the {:s} direction'.format(
                    max_peak_drift, max_peak_drift_loc,
                    max_peak_drift_dir))
            print(
                'Maximum peak absolute floor acceleration {:.4f} g '
                'at floor {:d} in the {:s} direction '
                '(0 = base)'.format(
                    max_peak_accel, max_peak_accel_loc,
                    max_peak_accel_dir))
            print('Total Hysteretic Energy: {:.6f} kN·m'.format(
                total_hysteretic_energy))
            for ei in range(n_elements):
                print('  Storey {:d} Hysteretic Energy: {:.6f} kN·m'.format(
                    ei + 1, hysteretic_energy_per_storey[ei]))

        #  Optional animation
        if save_animation_path is not None:
            try:
                print("\nGenerating NRHA animation...")
                time_array = np.arange(step) * dt_ansys
                acc_input_full = np.loadtxt(fnames[0])
                gm_time = np.arange(len(acc_input_full)) * dt_gm
                acc_resampled = np.interp(
                    time_array, gm_time, acc_input_full) / 9.81

                min_len = min(len(time_array), len(acc_resampled),
                              node_disps.shape[0], node_accels.shape[0])
                time_array = time_array[:min_len]
                acc_resampled = acc_resampled[:min_len]
                node_disps = node_disps[:min_len, :]
                node_accels = node_accels[:min_len, :]

                max_frames = 200
                frame_step = max(1, len(time_array) // max_frames)
                frames = np.arange(0, len(time_array), frame_step)

                pl = plotter()
                # Exact time MinMax/convergence failure was detected
                collapse_t = collapse_time
                pl.animate_nrha(control_nodes=control_nodes,
                                acc=acc_resampled[frames],
                                dts=time_array[frames],
                                nrha_disps=node_disps[frames, :],
                                nrha_accels=node_accels[frames, :],
                                drift_thresholds=drift_thresholds,
                                export_path=save_animation_path,
                                collapse_time=collapse_t,
                                true_peak_drift=peak_drift[:, 0],
                                true_peak_accel=peak_accel[:, 0])
            except Exception as e:
                print(f"Animation generation failed: {e}")

        #  Return outputs
        return (control_nodes, conv_index, peak_drift, peak_accel,
                max_peak_drift, max_peak_drift_dir, max_peak_drift_loc,
                max_peak_accel, max_peak_accel_dir, max_peak_accel_loc,
                peak_disp,
                hysteretic_energy_per_storey, total_hysteretic_energy)

    def do_ida_analysis(
        self,
        fnames,
        dt_gm,
        t_max,
        dt_ansys,
        target_drift=0.05,
        initial_sf=0.1,
        hunt_step=2.0,
        max_fill_gap=0.2,
        max_runs=15,
        capping_drift=0.10,
        xi=0.05,
        pFlag=False
    ):
        """
        Performs Incremental Dynamic Analysis (IDA) using the 'Hunt,
        Trace and Fill' algorithm as per Vamvatsikos and Cornell
        (2002, 2004).

        The algorithm first 'hunts' for the collapse capacity by
        increasing the scale factor (SF) geometrically. Once collapse
        or the target drift is reached, it 'traces' back with smaller
        steps for the scaling factor and 'fills' the gaps between
        successful runs to refine the IDA curve.

        Parameters
        ----------
        fnames : list
            List of file paths to the ground motion records for each direction.

        dt_gm : float
            Time-step of the ground motion records.

        t_max : float
            The maximum time duration for each individual analysis run.

        dt_ansys : float
            The integration time-step at which the structural
            analysis will be conducted.

        target_drift : float, optional, default=0.05
            The drift ratio threshold considered as structural
            collapse (e.g., 5%).

        initial_sf : float, optional, default=0.1
            The starting scale factor for the first simulation run.

        hunt_step : float, optional, default=2.0
            The geometric multiplier used to increase the scale
            factor during the 'Hunt' phase.

        max_fill_gap : float, optional, default=0.2
            The maximum allowable gap between scale factors. If a
            gap is larger, the 'Fill' phase will bisect it.

        max_runs : int, optional, default=15
            Maximum total number of nonlinear time-history
            simulations allowed.

        capping_drift : float, optional, default=0.10
            The drift value assigned to failed or collapsed runs
            for visualization (flatlining).

        xi : float, optional, default=0.05
            The damping ratio used in the analysis (default is 5%).

        Returns
        -------
        ida_data : dict
            A dictionary where keys are scale factors and values
            are dictionaries containing results (peak drift,
            acceleration, convergence state, etc.) for that run.

        ordered_sfs : list
            A list of all scale factors tested, in the order they
            were executed.

        Note
        ----
        The current method assumes the acceleration time-history is
        in m/s2. Therefore, the acceleration values are multiplied
        by a factor of g.

        References
        ----------
        [1] Vamvatsikos, D. and Cornell, C.A. (2002), Incremental
            dynamic analysis. Earthquake Engng. Struct. Dyn.,
            31: 491-514. https://doi.org/10.1002/eqe.141
        [2] Vamvatsikos D, Cornell CA. Applied Incremental Dynamic
            Analysis. Earthquake Spectra. 2004;20(2):523-553.
            doi:10.1193/1.1737737
        """

        ida_data = {}
        ordered_sfs = []
        self.run_count = 0

        def run_step(sf_value):
            """Execute a single simulation step and capture results."""
            if self.run_count >= max_runs:
                print(
                    f"Execution Limit Reached: {max_runs} runs. "
                    f"Skipping SF {sf_value:.3f}")
                return None, None
            print(
                f" -- Run {self.run_count + 1}/{max_runs}"
                f" | SF: {sf_value:.3f}"
            )

            # Reset environment and rebuild model for the current iteration
            ops.wipe()
            self.compile_model()
            self.do_gravity_analysis()

            # Execute the nonlinear time-history analysis
            res = self.do_nrha_analysis(
                fnames, dt_gm, sf_value * units.g, t_max, dt_ansys,
                pFlag=pFlag, xi=xi)

            # # Check convergence state and extract max drift
            raw_max_drift = res[4]
            conv_state = res[1]

            # Flatline check: cap drift if solver failed or drift
            # exceeds target (default cap = 0.10, i.e., 10% drift)
            if conv_state == -1 or raw_max_drift >= target_drift:
                final_drift = capping_drift
                print(
                    f"Collapse/Target Reached! Capping drift at {final_drift}")
            else:
                final_drift = raw_max_drift

            self.run_count += 1
            ordered_sfs.append(sf_value)

            # Store results in the main data dictionary
            ida_data[sf_value] = {'control_nodes': res[0],
                                  'conv_index': conv_state,
                                  'peak_drift': res[2],
                                  'peak_accel': res[3],
                                  'max_peak_drift': final_drift,
                                  'max_peak_drift_dir': res[5],
                                  'max_peak_drift_loc': res[6],
                                  'max_peak_accel': res[7],
                                  'max_peak_accel_dir': res[8],
                                  'max_peak_accel_loc': res[9],
                                  'peak_disp': res[10],
                                  'hysteretic_energy_per_storey': res[11],
                                  'total_hysteretic_energy': res[12]}
            return final_drift, conv_state

        # Phase 1: Let's go hunting for collapse!
        # Rapidly increase SF to find the building's limit
        curr_sf = initial_sf
        while self.run_count < max_runs:
            m_drift, state = run_step(curr_sf)
            if m_drift is None:
                break
            if state == -1 or m_drift >= target_drift:
                print(f"Hunt terminated: Limit reached at SF = {curr_sf}")
                break
            curr_sf *= hunt_step

        # Phase 2: Trace and fill to refine the IDA curve
        while self.run_count < max_runs:
            sorted_sfs = sorted(ida_data.keys())
            refined = False
            for i in range(len(sorted_sfs) - 1):
                if self.run_count >= max_runs:
                    break
                sf_low, sf_high = sorted_sfs[i], sorted_sfs[i + 1]

                # Only fill if not already in the flatline region
                if (sf_high - sf_low) > max_fill_gap and \
                        ida_data[sf_high]['max_peak_drift'] < 0.10:
                    mid_sf = (sf_low + sf_high) / 2.0
                    run_step(mid_sf)
                    refined = True

            if not refined or self.run_count >= max_runs:
                break

        return ida_data, ordered_sfs

    def do_nrha_analysis_sequences(
        self, fnames, time_vector, sf,
        pFlag=True, xi=0.05, ansys_soe='BandGeneral',
        constraints_handler='Plain', numberer='RCM',
        test_type='NormDispIncr', init_tol=1.0e-6, init_iter=50,
        algorithm_type='Newton', save_animation_path=None,
        drift_thresholds=None, padding_duration=40.0,
        quiescence_threshold=1.0e-6
    ):
        """
        Perform nonlinear time-history analysis on a
        Multi-Degree-of-Freedom (MDOF) system subjected to a
        concatenated sequence of ground-motion records that may
        have different time-steps and are separated by
        zero-acceleration padding intervals.

        Unlike ``do_nrha_analysis`` which takes a fixed scalar
        ``dt_gm`` and a constant ``dt_ansys``, this method takes
        an explicit ``time_vector`` so that records with different
        sampling rates can be stitched together and the analysis
        steps through each point at its native dt.

        The method automatically detects individual ground-motion
        records within the concatenated input by identifying
        padding (quiescent) zones where the absolute acceleration
        stays below *quiescence_threshold* for at least
        *padding_duration* seconds. Peak responses and hysteretic
        energies are reported both for the full sequence and for
        each individual record.

        Supports uni-directional (1 file) and bi-directional
        (2 files) ground-motion loading. Floor accelerations are
        recorded as absolute (total) accelerations, including at
        the base. Hysteretic energy dissipation is computed via
        signed force-velocity integration (trapezoidal rule),
        correctly capturing only dissipated energy and not elastic
        recovery.

        Parameters
        ----------
        fnames : list
            List of file paths to the ground-motion records. One
            file applies X-direction loading; two files apply
            bi-directional (X and Y) loading simultaneously. Each
            file contains a concatenated acceleration time-history
            (one value per time_vector entry) where individual
            records are separated by zero-padded intervals.

        time_vector : array-like
            Monotonically increasing time values (s) corresponding
            to each acceleration sample in the ground-motion
            files. This may be irregularly spaced when records
            with different dt are concatenated.

        sf : float
            Scale factor to apply to the ground-motion records.
            Typically equal to gravitational acceleration
            (9.81 m/s²) when records are in units of g.

        pFlag : bool, optional, default=True
            If True, prints progress updates during the analysis.

        xi : float, optional, default=0.05
            Inherent viscous damping ratio (default is 5%).

        ansys_soe : str, optional, default='BandGeneral'
            System of equations solver type.

        constraints_handler : str, optional, default='Plain'
            Method used to enforce constraint equations.

        numberer : str, optional, default='RCM'
            DOF numberer object (Reverse Cuthill-McKee by default).

        test_type : str, optional, default='NormDispIncr'
            Convergence test type.

        init_tol : float, optional, default=1.0e-6
            Convergence tolerance.

        init_iter : int, optional, default=50
            Maximum number of iterations per time step.

        algorithm_type : str, optional, default='Newton'
            Nonlinear solution algorithm.

        save_animation_path : str, optional
            If provided, saves the NRHA animation to this file
            path (e.g., 'nrha.gif').

        drift_thresholds : list, optional
            Drift thresholds used in the animation for
            damage-state colour changes.

        padding_duration : float, optional, default=40.0
            Minimum duration (s) of near-zero acceleration that
            separates two consecutive ground-motion records within
            the concatenated file. Used by the automatic sequence
            detector.

        quiescence_threshold : float, optional, default=1.0e-6
            Absolute acceleration amplitude below which a sample
            is considered 'silent'. Used together with
            *padding_duration* to locate record boundaries.

        Returns
        -------
        control_nodes : list
            Node tags (base first, then floors).

        conv_index : int
            Convergence status: 0 = success, -1 = failure.

        peak_drift : np.ndarray
            Peak inter-storey drift ratio per storey,
            shape (n_storeys, 2). Col 0 = X, col 1 = Y.
            Full sequence.

        peak_accel : np.ndarray
            Peak absolute floor acceleration (g) per node
            (base + floors), shape (n_nodes, 2). Col 0 = X,
            col 1 = Y. Full sequence.

        max_peak_drift : float
            Maximum peak inter-storey drift ratio across all
            storeys and directions (full sequence).

        max_peak_drift_dir : str
            Direction ('X' or 'Y') of the maximum peak drift.

        max_peak_drift_loc : int
            Storey number (1-based) of the maximum peak drift.

        max_peak_accel : float
            Maximum peak absolute floor acceleration (g).

        max_peak_accel_dir : str
            Direction ('X' or 'Y') of the maximum peak accel.

        max_peak_accel_loc : int
            Floor number (0-based, 0 = base) of the maximum
            peak acceleration.

        peak_disp : np.ndarray
            Peak relative displacement (m) per node,
            shape (n_nodes, 2).

        hysteretic_energy_per_storey : np.ndarray
            Dissipated hysteretic energy per storey (kN·m),
            shape (n_storeys,). Full sequence.

        total_hysteretic_energy : float
            Total dissipated hysteretic energy (kN·m).
            Full sequence.

        hysteretic_energy_per_storey_per_sequence :
            list of np.ndarray
            Per-sequence dissipated hysteretic energy per
            storey (kN·m), each of shape (n_storeys,).
            For a two-record sequence these correspond to the
            original G1 and G2 outputs.

        total_hysteretic_energy_per_sequence : list of float
            Per-sequence total dissipated hysteretic energy
            (kN·m).

        max_peak_drift_per_sequence : list of float
            Per-sequence maximum peak drift ratio.

        peak_drift_per_sequence : list of np.ndarray
            Per-sequence peak inter-storey drift ratio,
            each of shape (n_storeys, 2).

        n_sequences : int
            Number of individual ground-motion records
            detected in the concatenated file.

        sequence_boundaries : list of tuple
            List of (t_start, t_end) pairs (in seconds)
            defining the time window of each detected
            ground-motion record (excluding padding).
        """

        time_vector = np.asarray(time_vector, dtype=float)

        # --------------------------------------------------------
        #  Align with the '-dt' / '-filePath' convention used by
        #  do_nrha_analysis.
        #
        #  '-dt' form:  acc[i] is placed at t = i * dt.
        #    So for n values: t = [0, dt, 2dt, ..., (n-1)*dt]
        #
        #  User-supplied time vectors typically start at dt:
        #    tv = [dt, 2dt, ..., n*dt]   (n entries)
        #
        #  To replicate the -dt mapping with -time/-values,
        #  we need: t = [0, dt, 2dt, ..., (n-1)*dt]
        #  which is  [0, tv[0], tv[1], ..., tv[n-2]]
        #  i.e. prepend 0 and drop the last time entry.
        #  The acc array stays unchanged (n values).
        # --------------------------------------------------------
        acc_x_full = np.loadtxt(fnames[0])
        # Save the original last time value BEFORE any shift,
        # so the loop termination matches do_nrha_analysis exactly.
        t_max_original = float(time_vector[-1])
        prepended = time_vector[0] > 0.0
        if prepended:
            time_vector = np.concatenate(
                ([0.0], time_vector[:-1]))

        n_time_pts = len(time_vector)

        # Verify lengths match
        if len(acc_x_full) != n_time_pts:
            raise ValueError(
                f"Length of X-direction acceleration file "
                f"({len(acc_x_full)}) does not match "
                f"time_vector length ({n_time_pts}). "
                f"prepended={prepended}")

        # Pre-compute the dt for every analysis step from the time
        # vector.  n_analysis_steps = n_time_pts - 1 (the intervals
        # between consecutive time points), which matches the number
        # of ops.analyze() calls that do_nrha_analysis makes for
        # the same record when dt_ansys == dt_gm.
        dt_steps = np.diff(time_vector)
        n_analysis_steps = len(dt_steps)

        # A sample is 'silent' when it is exactly zero or its
        # absolute value is below quiescence_threshold.  The
        # original threshold-only approach missed records whose
        # tails decay to small-but-nonzero values (e.g. 1e-4).
        # Using exact-zero detection for the padding zone is more
        # robust because the padding is always written as 0.0.
        is_silent = np.abs(acc_x_full) <= quiescence_threshold

        # Locate contiguous silent blocks and record their
        # start/end indices and durations.  A block qualifies as
        # a padding separator when its duration is at least
        # padding_duration (with a small tolerance to absorb
        # floating-point drift in the time vector).
        pad_tol = 0.5  # seconds — tolerance on padding_duration
        silent_blocks = []  # list of (start_idx, end_idx)
        block_start = None

        for idx in range(n_time_pts):
            if is_silent[idx]:
                if block_start is None:
                    block_start = idx
            else:
                if block_start is not None:
                    block_end = idx - 1  # last silent sample
                    block_dur = (time_vector[block_end]
                                 - time_vector[block_start])
                    if block_dur >= (padding_duration - pad_tol):
                        silent_blocks.append(
                            (block_start, block_end))
                    block_start = None

        # Handle a trailing silent block (zeros at the very end)
        if block_start is not None:
            block_end = n_time_pts - 1
            block_dur = (time_vector[block_end]
                         - time_vector[block_start])
            if block_dur >= (padding_duration - pad_tol):
                silent_blocks.append((block_start, block_end))

        # Build record boundaries from the gaps between (and
        # around) silent blocks.  Each active record spans from
        # the first non-silent sample after the previous padding
        # to the last non-silent sample before the next padding.
        boundaries = []
        rec_start_idx = 0  # start of first possible record

        # Skip any leading silent block (record starts after it)
        if (silent_blocks
                and silent_blocks[0][0] == 0):
            rec_start_idx = silent_blocks[0][1] + 1
            silent_blocks = silent_blocks[1:]

        # Skip any trailing silent block (not a separator)
        if (silent_blocks
                and silent_blocks[-1][1] == n_time_pts - 1):
            trailing_end = silent_blocks[-1][0] - 1
            silent_blocks = silent_blocks[:-1]
        else:
            trailing_end = n_time_pts - 1

        if not silent_blocks:
            # No interior padding found — single record
            boundaries.append((
                time_vector[rec_start_idx],
                time_vector[trailing_end]))
        else:
            for sb_start, sb_end in silent_blocks:
                # Record before this padding block
                rec_end_idx = sb_start - 1
                if rec_end_idx >= rec_start_idx:
                    boundaries.append((
                        time_vector[rec_start_idx],
                        time_vector[rec_end_idx]))
                # Next record starts after this padding block
                rec_start_idx = sb_end + 1

            # Record after the last padding block
            if rec_start_idx <= trailing_end:
                boundaries.append((
                    time_vector[rec_start_idx],
                    time_vector[trailing_end]))

        n_sequences = len(boundaries)
        if pFlag:
            print(f'Detected {n_sequences} ground-motion record(s) '
                  f'in the concatenated input.')
            for seq_i, (ts, te) in enumerate(boundaries):
                print(f'  Record {seq_i + 1}: '
                      f'{ts:.2f} – {te:.2f} s')

        # --------------------------------------------------------
        #  MinMax deformation limits
        # --------------------------------------------------------
        minmax_limits = 1.0 * np.abs(self.storey_drifts[:, -1])

        # --------------------------------------------------------
        #  Determine loading directions
        # --------------------------------------------------------
        bidir = len(fnames) >= 2

        # Define control nodes
        control_nodes = ops.getNodeTags()
        n_nodes = len(control_nodes)

        # --------------------------------------------------------
        #  Apply ground-motion time-series & uniform excitation
        #  Using '-time' + '-values' form so that irregularly
        #  spaced time vectors are handled correctly.
        # --------------------------------------------------------
        if len(fnames) > 0:
            ops.timeSeries(
                'Path', 1,
                '-time', *time_vector,
                '-values', *acc_x_full,
                '-factor', sf)
            ops.pattern('UniformExcitation', 1, 1, '-accel', 1)
        if len(fnames) > 1:
            acc_y_full = np.loadtxt(fnames[1])
            if len(acc_y_full) != n_time_pts:
                raise ValueError(
                    f"Length of Y-direction acceleration file "
                    f"({len(acc_y_full)}) does not match "
                    f"time_vector length ({n_time_pts}).")
            ops.timeSeries(
                'Path', 2,
                '-time', *time_vector,
                '-values', *acc_y_full,
                '-factor', sf)
            ops.pattern('UniformExcitation', 2, 2, '-accel', 2)
        if len(fnames) > 2:
            acc_z_full = np.loadtxt(fnames[2])
            if len(acc_z_full) != n_time_pts:
                raise ValueError(
                    f"Length of Z-direction acceleration file "
                    f"({len(acc_z_full)}) does not match "
                    f"time_vector length ({n_time_pts}).")
            ops.timeSeries(
                'Path', 3,
                '-time', *time_vector,
                '-values', *acc_z_full,
                '-factor', sf)
            ops.pattern('UniformExcitation', 3, 3, '-accel', 3)

        # --------------------------------------------------------
        #  Configure analysis objects
        # --------------------------------------------------------
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)
        ops.algorithm(algorithm_type)
        ops.integrator('Newmark', 0.5, 0.25)
        ops.analysis('Transient')

        # Analysis state
        conv_index = 0
        collapse_time = None
        control_time = 0.0
        ok = 0

        # --------------------------------------------------------
        #  Storey height array for IDR normalisation
        # --------------------------------------------------------
        if n_nodes < 2:
            top_nodes = []
            bottom_nodes = []
        else:
            top_nodes = control_nodes[1:]
            bottom_nodes = control_nodes[:-1]

        h = []
        for i in range(len(top_nodes)):
            topZ = ops.nodeCoord(top_nodes[i], 3)
            bottomZ = ops.nodeCoord(bottom_nodes[i], 3)
            dist = topZ - bottomZ
            if dist == 0:
                print(
                    "WARNING: Zero storey height detected "
                    "— using 1e9 to avoid division by zero.")
                h.append(1e9)
            else:
                h.append(dist)
        h = np.array(h) if len(h) > 0 else np.array([])

        # --------------------------------------------------------
        #  Pre-allocate recording arrays
        #  Allocate n_time_pts + 2 rows to accommodate the extra
        #  step(s) that the while control_time <= t_max_loop
        #  termination may run beyond the time vector range
        #  (matching do_nrha_analysis's behaviour).
        # --------------------------------------------------------
        n_storeys = len(top_nodes)
        n_alloc = n_time_pts + 2

        # Relative displacements (X and Y separately)
        node_disps_x = np.zeros((n_alloc, n_nodes))
        node_disps_y = np.zeros((n_alloc, n_nodes))

        # Absolute (total) accelerations in g
        node_accels_x = np.zeros((n_alloc, n_nodes))
        node_accels_y = np.zeros((n_alloc, n_nodes))

        # Peak trackers — shape (n_nodes, 2): col 0 = X, col 1 = Y
        peak_disp = np.zeros((n_nodes, 2))
        peak_accel = np.zeros((n_nodes, 2))

        # Peak IDR — shape (n_storeys, 2)
        peak_drift = np.zeros((n_storeys, 2))

        # Per-sequence peak drift trackers
        peak_drift_per_seq = [
            np.zeros((n_storeys, 2)) for _ in range(n_sequences)
        ]

        # --------------------------------------------------------
        #  Rayleigh damping
        # --------------------------------------------------------
        num_frequencies = len(self.omega)
        if num_frequencies == 1:
            alphaM = 2 * self.omega[0] * xi
            ops.rayleigh(alphaM, 0, 0, 0)
        elif num_frequencies >= 2:
            idx_high = min(num_frequencies - 1, 2)
            alphaM = (2 * self.omega[0] * self.omega[idx_high]
                      * xi
                      / (self.omega[0] + self.omega[idx_high]))
            alphaK = (2 * xi
                      / (self.omega[0] + self.omega[idx_high]))
            ops.rayleigh(alphaM, 0, alphaK, 0)

        # --------------------------------------------------------
        #  Hysteretic energy tracking (signed F·v trapezoidal)
        # --------------------------------------------------------
        element_tags = ops.getEleTags()
        n_elements = len(element_tags)

        energy_force_prev_x = np.zeros(n_elements)
        energy_force_prev_y = np.zeros(n_elements)
        energy_vel_prev_x = np.zeros(n_elements)
        energy_vel_prev_y = np.zeros(n_elements)
        hysteretic_energy_per_storey = np.zeros(n_elements)
        energy_time_prev = 0.0

        # Per-sequence energy accumulators
        hysteretic_energy_per_storey_per_seq = [
            np.zeros(n_elements) for _ in range(n_sequences)
        ]

        # Progress print throttle
        print_every = max(1, int(np.ceil(n_analysis_steps / 50.0)))

        # --------------------------------------------------------
        #  Helper: determine which sequence index a time belongs to
        # --------------------------------------------------------
        def _get_sequence_index(t):
            """Return the 0-based sequence index for time *t*,
            or -1 if *t* falls in a padding zone."""
            for si, (ts, te) in enumerate(boundaries):
                if ts <= t <= te:
                    return si
            return -1

        # --------------------------------------------------------
        #  Main time-stepping loop
        #  Uses the same termination condition as do_nrha_analysis
        #  (control_time <= t_max) so that both methods execute
        #  exactly the same number of analysis steps.  The dt for
        #  each step comes from dt_steps while within the time
        #  vector range; beyond that, the last dt is reused (this
        #  covers the extra free-vibration step(s) that
        #  do_nrha_analysis performs when t_max > time_vector[-1]).
        # --------------------------------------------------------
        # Use the original (un-shifted) t_max so the loop runs
        # the same duration as do_nrha_analysis.
        t_max_loop = t_max_original

        step = 0
        while (conv_index == 0
               and control_time <= t_max_loop
               and ok == 0):

            if step < n_analysis_steps:
                dt_current = dt_steps[step]
            else:
                dt_current = dt_steps[-1]  # reuse last dt

            ok = ops.analyze(1, dt_current)
            control_time = ops.getTime()

            if pFlag and (step % print_every == 0
                          or control_time >= t_max_loop):
                print(f'Step {step + 1}: '
                      f'Completed {control_time:.3f} of '
                      f'{t_max_loop:.3f} seconds')

            # ---- Adaptive convergence recovery ----
            if ok != 0:
                print('FAILED at {:.3f}: Trying half '
                      'time-step.'.format(control_time))
                ok = ops.analyze(1, 0.5 * dt_current)
            if ok != 0:
                print('FAILED at {:.3f}: Trying quarter '
                      'time-step.'.format(control_time))
                ok = ops.analyze(1, 0.25 * dt_current)
            if ok != 0:
                print('FAILED at {:.3f}: Relaxing convergence '
                      '+ more iterations.'.format(control_time))
                ops.test(test_type, init_tol * 0.01,
                         init_iter * 10)
                ok = ops.analyze(1, 0.5 * dt_current)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                print('FAILED at {:.3f}: Newton '
                      'initialThenCurrent.'.format(control_time))
                ops.test(test_type, init_tol * 0.01,
                         init_iter * 10)
                ops.algorithm('Newton', 'initialThenCurrent')
                ok = ops.analyze(1, 0.5 * dt_current)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                print('FAILED at {:.3f}: Newton '
                      'initial.'.format(control_time))
                ops.test(test_type, init_tol * 0.01,
                         init_iter * 10)
                ops.algorithm('Newton', 'initial')
                ok = ops.analyze(1, 0.5 * dt_current)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                print('FAILED at {:.3f}: Hail Mary '
                      '(FixedNumIter).'.format(control_time))
                ops.test('FixedNumIter', init_iter * 10)
                ok = ops.analyze(1, 0.5 * dt_current)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                print('FAILED at {:.3f}: Exiting '
                      'analysis.'.format(control_time))
                conv_index = -1
                collapse_time = control_time
                break

            # ---- Record nodal responses ----
            for i, node in enumerate(control_nodes):
                disp_x = ops.nodeDisp(node, 1)
                disp_y = ops.nodeDisp(node, 2)
                node_disps_x[step, i] = disp_x
                node_disps_y[step, i] = disp_y

                if abs(disp_x) > peak_disp[i, 0]:
                    peak_disp[i, 0] = abs(disp_x)
                if abs(disp_y) > peak_disp[i, 1]:
                    peak_disp[i, 1] = abs(disp_y)

                ag_x = (ops.getLoadFactor(1)
                        if len(fnames) > 0 else 0.0)
                ag_y = (ops.getLoadFactor(2)
                        if len(fnames) > 1 else 0.0)
                abs_accel_x = (
                    (ops.nodeAccel(node, 1) + ag_x) / 9.81)
                abs_accel_y = (
                    (ops.nodeAccel(node, 2) + ag_y) / 9.81)
                node_accels_x[step, i] = abs_accel_x
                node_accels_y[step, i] = abs_accel_y

                if abs(abs_accel_x) > peak_accel[i, 0]:
                    peak_accel[i, 0] = abs(abs_accel_x)
                if abs(abs_accel_y) > peak_accel[i, 1]:
                    peak_accel[i, 1] = abs(abs_accel_y)

            # ---- Inter-storey drift ratios ----
            seq_idx = _get_sequence_index(control_time)

            if n_storeys > 0:
                dx_top = node_disps_x[step, 1:]
                dx_bot = node_disps_x[step, :-1]
                idr_x = np.abs(dx_top - dx_bot) / h
                mask_x = idr_x > peak_drift[:, 0]
                peak_drift[mask_x, 0] = idr_x[mask_x]

                dy_top = node_disps_y[step, 1:]
                dy_bot = node_disps_y[step, :-1]
                idr_y = np.abs(dy_top - dy_bot) / h
                mask_y = idr_y > peak_drift[:, 1]
                peak_drift[mask_y, 1] = idr_y[mask_y]

                # Per-sequence peak drift update
                if seq_idx >= 0:
                    pd_seq = peak_drift_per_seq[seq_idx]
                    mask_sx = idr_x > pd_seq[:, 0]
                    pd_seq[mask_sx, 0] = idr_x[mask_sx]
                    mask_sy = idr_y > pd_seq[:, 1]
                    pd_seq[mask_sy, 1] = idr_y[mask_sy]

            # ---- Hysteretic energy (signed F·v trapezoidal) ----
            # Same sign convention as do_nrha_analysis: use
            # eleForce[0] directly (node-I force) so that both
            # methods produce identical energy values.
            energy_time_curr = control_time
            dt_energy = energy_time_curr - energy_time_prev
            if dt_energy > 0:
                for ei, ele_tag in enumerate(element_tags):
                    ele_force_vec = ops.eleForce(ele_tag)
                    force_curr_x = ele_force_vec[0]
                    force_curr_y = ele_force_vec[1]

                    node_top = control_nodes[ei + 1]
                    node_bot = control_nodes[ei]

                    vel_curr_x = (ops.nodeVel(node_top, 1)
                                  - ops.nodeVel(node_bot, 1))
                    vel_curr_y = (ops.nodeVel(node_top, 2)
                                  - ops.nodeVel(node_bot, 2))

                    power_prev = (
                        energy_force_prev_x[ei]
                        * energy_vel_prev_x[ei]
                        + energy_force_prev_y[ei]
                        * energy_vel_prev_y[ei]
                    )
                    power_curr = (force_curr_x * vel_curr_x
                                  + force_curr_y * vel_curr_y)
                    dE = 0.5 * (power_prev + power_curr) * dt_energy
                    hysteretic_energy_per_storey[ei] += dE

                    # Accumulate into the active sequence
                    if seq_idx >= 0:
                        hysteretic_energy_per_storey_per_seq[
                            seq_idx][ei] += dE

                    # Update previous-step values
                    energy_force_prev_x[ei] = force_curr_x
                    energy_force_prev_y[ei] = force_curr_y
                    energy_vel_prev_x[ei] = vel_curr_x
                    energy_vel_prev_y[ei] = vel_curr_y

            energy_time_prev = energy_time_curr
            step += 1

            # ---- MinMax spring failure check ----
            for s_idx, ele in enumerate(element_tags):
                try:
                    deform_result = ops.eleResponse(
                        ele, 'deformation')
                    if deform_result is None:
                        if pFlag:
                            print(
                                f'COLLAPSE DETECTED: Spring at '
                                f'storey {s_idx + 1} killed by '
                                f'MinMax at '
                                f't={control_time:.3f}s.')
                        conv_index = -1
                        collapse_time = control_time
                        break
                    spring_deform = abs(deform_result[0])
                    if spring_deform >= minmax_limits[s_idx]:
                        if pFlag:
                            print(
                                f'COLLAPSE DETECTED! Spring at '
                                f'storey {s_idx + 1} reached '
                                f'MinMax limit '
                                f'({spring_deform:.4f} >= '
                                f'{minmax_limits[s_idx]:.4f}) '
                                f'at t={control_time:.3f}s. '
                                f'Capping drift and terminating.')
                        conv_index = -1
                        collapse_time = control_time
                        break
                except Exception:
                    if pFlag:
                        print(
                            f'COLLAPSE DETECTED! Spring at '
                            f'storey {s_idx + 1} unresponsive '
                            f'(MinMax killed) at '
                            f't={control_time:.3f}s.')
                    conv_index = -1
                    collapse_time = control_time
                    break
            if conv_index == -1:
                break

        # --------------------------------------------------------
        #  Trim arrays to actual number of completed steps
        # --------------------------------------------------------
        node_disps_x = node_disps_x[:step, :]
        node_disps_y = node_disps_y[:step, :]
        node_accels_x = node_accels_x[:step, :]
        node_accels_y = node_accels_y[:step, :]

        node_disps = node_disps_x.copy()
        node_accels = node_accels_x.copy()

        # --------------------------------------------------------
        #  Maximum drift summary (full sequence)
        # --------------------------------------------------------
        max_peak_drift = (np.max(peak_drift)
                          if peak_drift.size > 0 else 0.0)
        if peak_drift.size > 0:
            ind = np.unravel_index(
                np.argmax(peak_drift), peak_drift.shape)
            max_peak_drift_dir = 'X' if ind[1] == 0 else 'Y'
            max_peak_drift_loc = ind[0] + 1
        else:
            max_peak_drift_dir = 'X'
            max_peak_drift_loc = 0

        # Per-sequence max drift
        max_peak_drift_per_seq = []
        for pd_seq in peak_drift_per_seq:
            max_peak_drift_per_seq.append(
                float(np.max(pd_seq)) if pd_seq.size > 0
                else 0.0
            )

        # --------------------------------------------------------
        #  Maximum acceleration summary
        # --------------------------------------------------------
        max_peak_accel = np.max(peak_accel)
        if peak_accel.size > 0:
            ind_a = np.unravel_index(
                np.argmax(peak_accel), peak_accel.shape)
            max_peak_accel_dir = 'X' if ind_a[1] == 0 else 'Y'
            max_peak_accel_loc = ind_a[0]
        else:
            max_peak_accel_dir = 'X'
            max_peak_accel_loc = 0

        # --------------------------------------------------------
        #  Total hysteretic energy
        # --------------------------------------------------------
        total_hysteretic_energy = float(
            np.sum(hysteretic_energy_per_storey))

        total_hysteretic_energy_per_seq = [
            float(np.sum(e))
            for e in hysteretic_energy_per_storey_per_seq
        ]

        # --------------------------------------------------------
        #  Console feedback
        # --------------------------------------------------------
        if conv_index == -1:
            print('------ ANALYSIS FAILED --------')
        else:
            print('~~~~~~~ ANALYSIS SUCCESSFUL ~~~~~~~~~')

        if pFlag:
            direction_label = (
                'bi-directional (X+Y)' if bidir
                else 'uni-directional (X)')
            print(f'Loading: {direction_label}')
            print(
                'Final state = {:d} (-1 for non-converged, '
                '0 for stable)'.format(conv_index))
            print(
                'Maximum peak storey drift {:.4f} at storey {:d} '
                'in the {:s} direction'.format(
                    max_peak_drift, max_peak_drift_loc,
                    max_peak_drift_dir))
            print(
                'Maximum peak absolute floor acceleration '
                '{:.4f} g at floor {:d} in the {:s} direction '
                '(0 = base)'.format(
                    max_peak_accel, max_peak_accel_loc,
                    max_peak_accel_dir))
            print('Total Hysteretic Energy: {:.6f} kN·m'.format(
                total_hysteretic_energy))
            for ei in range(n_elements):
                print(
                    '  Storey {:d} Hysteretic Energy: '
                    '{:.6f} kN·m'.format(
                        ei + 1,
                        hysteretic_energy_per_storey[ei]))
            # Per-sequence summary
            for si in range(n_sequences):
                ts, te = boundaries[si]
                print(
                    f'\n  --- Record {si + 1} '
                    f'({ts:.2f}–{te:.2f} s) ---')
                print(
                    f'  Max peak drift: '
                    f'{max_peak_drift_per_seq[si]:.4f}')
                print(
                    f'  Total hysteretic energy: '
                    f'{total_hysteretic_energy_per_seq[si]:.6f}'
                    f' kN·m')
                for ei in range(n_elements):
                    print(
                        f'    Storey {ei + 1}: '
                        f'{hysteretic_energy_per_storey_per_seq[si][ei]:.6f}'
                        f' kN·m')

        # --------------------------------------------------------
        #  Optional animation
        # --------------------------------------------------------
        if save_animation_path is not None:
            try:
                print("\nGenerating NRHA animation...")
                time_array = time_vector[:step]
                acc_resampled = acc_x_full[:step] / 9.81

                min_len = min(
                    len(time_array), len(acc_resampled),
                    node_disps.shape[0], node_accels.shape[0])
                time_array = time_array[:min_len]
                acc_resampled = acc_resampled[:min_len]
                node_disps = node_disps[:min_len, :]
                node_accels = node_accels[:min_len, :]

                max_frames = 200
                frame_step = max(1, len(time_array) // max_frames)
                frames = np.arange(
                    0, len(time_array), frame_step)

                pl = plotter()
                collapse_t = collapse_time
                pl.animate_nrha(
                    control_nodes=control_nodes,
                    acc=acc_resampled[frames],
                    dts=time_array[frames],
                    nrha_disps=node_disps[frames, :],
                    nrha_accels=node_accels[frames, :],
                    drift_thresholds=drift_thresholds,
                    export_path=save_animation_path,
                    collapse_time=collapse_t,
                    true_peak_drift=peak_drift[:, 0],
                    true_peak_accel=peak_accel[:, 0])
            except Exception as e:
                print(f"Animation generation failed: {e}")

        # --------------------------------------------------------
        #  Return outputs
        # --------------------------------------------------------
        #
        # The return tuple preserves positional compatibility with
        # the original modeller_withenergycalc.py output:
        #
        #   Index  Name
        #   -----  ----
        #    0     control_nodes
        #    1     conv_index
        #    2     peak_drift                        (n_storeys, 2)
        #    3     peak_accel                        (n_nodes, 2)
        #    4     max_peak_drift                    float
        #    5     max_peak_drift_dir                str
        #    6     max_peak_drift_loc                int (1-based)
        #    7     max_peak_accel                    float
        #    8     max_peak_accel_dir                str
        #    9     max_peak_accel_loc                int (0-based)
        #   10     peak_disp                         (n_nodes, 2)
        #   11     hysteretic_energy_per_storey      (n_elements,)
        #   12     total_hysteretic_energy            float
        #   13     hysteretic_energy_per_storey_per_sequence
        #                                list of (n_elements,) arrays
        #   14     total_hysteretic_energy_per_sequence
        #                                            list of float
        #   15     max_peak_drift_per_sequence        list of float
        #   16     peak_drift_per_sequence
        #                                list of (n_storeys, 2) arrays
        #   17     n_sequences                        int
        #   18     sequence_boundaries    list of (t_start, t_end)
        #
        return (control_nodes,
                conv_index,
                peak_drift,
                peak_accel,
                max_peak_drift,
                max_peak_drift_dir,
                max_peak_drift_loc,
                max_peak_accel,
                max_peak_accel_dir,
                max_peak_accel_loc,
                peak_disp,
                hysteretic_energy_per_storey,
                total_hysteretic_energy,
                hysteretic_energy_per_storey_per_seq,
                total_hysteretic_energy_per_seq,
                max_peak_drift_per_seq,
                peak_drift_per_seq,
                n_sequences,
                boundaries)
