
import os
import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops

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


    def plot_model(self, display_info=True):
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
                          solver = '-genBandArpack',
                          doRayleigh=False,
                          pflag=False):
        """
        Perform modal analysis on a multi-degree-of-freedom (MDOF) system to determine its natural frequencies
        and mode shapes.

        This method calculates the natural frequencies and corresponding mode shapes of the system. The natural
        frequencies are determined by solving the eigenvalue problem, and the mode shapes are normalized
        for the system's degrees of freedom. The results can be used to assess the dynamic characteristics
        of the system.

        Parameters
        ----------
        num_modes: int, optional
            The number of modes to consider in the analysis. Default is 3. This parameter determines how many
            modes will be computed in the modal analysis.

        solver: string, optional
            The type of solver to use for the eigenvalue problem. Default is '-genBandArpack', which uses a
            generalized banded Arnoldi method for large sparse eigenvalue problems.

        doRayleigh: bool, optional
            Flag to enable or disable Rayleigh damping in the modal analysis. This parameter is not used directly
            in this method but can be set in the OpenSees model. Default is False.

        pflag: bool, optional
            Flag to control whether to print the modal analysis report. If True, the fundamental period and
            mode shape will be printed to the console. Default is False.

        Returns
        -------
        T: array
            The periods of vibration for the system, calculated as 2π/ω, where ω are the natural frequencies
            obtained from the eigenvalue problem.

        mode_shape: list
            A list of the normalized mode shapes for the system, with each element representing the displacement
            in the x-direction for the corresponding mode. The mode shapes are normalized by the last node's
            displacement.
        """

        ### Get frequency and period
        self.omega = np.power(ops.eigen(solver, num_modes), 0.5)
        T = 2.0*np.pi/self.omega

        mode_shape = []
        # Extract mode shapes for all nodes (displacements in x)
        for k in range(1, self.number_storeys+1):
            ux = ops.nodeEigenvector(k, 1, 1)  # Displacement in x-direction
            mode_shape.append(ux)

        # Normalize the mode shape
        mode_shape = np.array(mode_shape)/mode_shape[-1]

        ### Print optional report
        if pflag:
            ops.modalProperties('-print')
            ### Print output
            print(r'Fundamental Period:  T = {:.3f} s'.format(T[0]))
            print('Mode Shape:', mode_shape)

        ### Wipe the analysis objects
        ops.wipeAnalysis()

        return T, mode_shape

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
                        algorithm_type='KrylovNewton'):
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
        spo_disps: array
            Displacements at each floor level during the pushover analysis.

        spo_rxn: array
            Base shear recorded as the sum of reactions at the base during the pushover analysis.

        spo_disps_spring: array
            Displacements in the storey zero-length elements (non-linear springs).

        spo_forces_spring: array
            Shear forces in the storey zero-length elements (non-linear springs).

        """

        # apply the load pattern
        ops.timeSeries("Linear", 1) # create timeSeries
        ops.pattern("Plain", 1, 1) # create a plain load pattern

        # define control nodes
        nodeList = ops.getNodeTags()
        control_node = nodeList[-1]
        pattern_nodes = nodeList[1:]
        rxn_nodes = [nodeList[0]]


        # we can integrate modal patterns, inverse triangular, etc.
        for i in np.arange(len(pattern_nodes)):
            if push_dir == 1:
                if len(pattern_nodes)==1:
                    ops.load(pattern_nodes[i], 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                else:
                    ops.load(pattern_nodes[i], phi[i]*self.floor_masses[i], 0.0, 0.0, 0.0, 0.0, 0.0) ######### IT STARTS FROM ZERO

            elif push_dir == 2:
                if len(pattern_nodes)==1:
                    ops.load(pattern_nodes[i], 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
                else:
                    ops.load(pattern_nodes[i], 0.0, phi[i]*self.floor_masses[i], 0.0, 0.0, 0.0, 0.0)

            elif push_dir == 3:
                if len(pattern_nodes)==1:
                    ops.load(pattern_nodes[i], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
                else:
                    ops.load(pattern_nodes[i], 0.0, 0.0, phi[i]*self.floor_masses[i], 0.0, 0.0, 0.0)

        # Set up the initial objects
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)
        ops.algorithm(algorithm_type)

        # Set the integrator
        target_disp = float(ref_disp)*float(disp_scale_factor)
        delta_disp = target_disp/(1.0*num_steps)
        ops.integrator('DisplacementControl', control_node, push_dir, delta_disp)
        ops.analysis('Static')

        # Get a list of all the element tags (zero-length springs)
        elementList = ops.getEleTags()

        # Give some feedback if requested
        if pflag is True:
            print(f"\n------ Static Pushover Analysis of Node # {control_node} to {target_disp} ---------")
        # Set up the analysis
        ok = 0
        step = 1
        loadf = 1.0

        # Recording base shear
        spo_rxn = np.array([0.])
        # Recording top displacement
        spo_top_disp = np.array([ops.nodeResponse(control_node, push_dir,1)])
        # Recording all displacements to estimate drifts
        spo_disps = np.array([[ops.nodeResponse(node, push_dir, 1) for node in pattern_nodes]])

        # Recording displacements and forces in non-linear zero-length springs [the zero is needed to get the exact required value]
        spo_disps_spring = np.array([[ops.eleResponse(ele, 'deformation')[0] for ele in elementList]])
        spo_forces_spring = np.array([[ops.eleResponse(ele, 'force')[0] for ele in elementList]])


        # Start the adaptive convergence scheme
        while step <= num_steps and ok == 0 and loadf > 0:

            # Push it by one step
            ok = ops.analyze(1)

            # If the analysis fails, try the following changes to achieve convergence
            if ok != 0:
                print('FAILED: Trying relaxing convergence...')
                ops.test(test_type, init_tol*0.01, init_iter)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                print('FAILED: Trying relaxing convergence with more iterations...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
            if ok != 0:
                print('FAILED: Trying relaxing convergence with more iteration and Newton with initial then current...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ops.algorithm('Newton', 'initialThenCurrent')
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                print('FAILED: Trying relaxing convergence with more iteration and Newton with initial...')
                ops.test(test_type, init_tol*0.01, init_iter*10)
                ops.algorithm('Newton', 'initial')
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)
                ops.algorithm(algorithm_type)
            if ok != 0:
                print('FAILED: Attempting a Hail Mary...')
                ops.test('FixedNumIter', init_iter*10)
                ok = ops.analyze(1)
                ops.test(test_type, init_tol, init_iter)

            # This feature of disabling the possibility of having a negative loading has been included.
            loadf = ops.getTime()

            # Give some feedback if requested
            if pflag is True:
                curr_disp = ops.nodeDisp(control_node, push_dir)
                print('Currently pushed node ' + str(control_node) + ' to ' + str(curr_disp) + ' with ' + str(loadf))

            # Increment to the next step
            step += 1

            # Get the results
            spo_top_disp = np.append(spo_top_disp, ops.nodeResponse(
            control_node, push_dir, 1))

            spo_disps = np.append(spo_disps, np.array([
            [ops.nodeResponse(node, push_dir, 1) for node in pattern_nodes]
            ]), axis=0)


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


        # Give some feedback on what happened
        if ok != 0:
            print('------ ANALYSIS FAILED --------')
        elif ok == 0:
            print('~~~~~~~ ANALYSIS SUCCESSFUL ~~~~~~~~~')
        if loadf < 0:
            print('Stopped because of load factor below zero')

        ### Wipe the analysis objects
        ops.wipeAnalysis()

        return spo_disps, spo_rxn, spo_disps_spring, spo_forces_spring

    def do_cpo_analysis(self,
                        ref_disp,
                        mu_levels,
                        push_dir,
                        dispIncr,
                        phi,
                        pflag=True,
                        num_steps=200,
                        ansys_soe='BandGeneral',
                        constraints_handler='Transformation',
                        numberer='RCM',
                        test_type='NormDispIncr',
                        init_tol=1.0e-5,
                        init_iter=1000,
                        algorithm_type='KrylovNewton'):
        """
        Perform cyclic pushover (CPO) analysis on a Multi-Degree-of-Freedom (MDOF) system.

        This method performs a cyclic pushover analysis where the structure is subjected
        to a series of incremental displacements in the specified direction, both positive
        and negative, to simulate cyclic loading (e.g., earthquake-like loading conditions).
        The pushover analysis is carried out over a specified number of cycles, with each cycle
        involving displacement increments to achieve the desired ductility.

        Parameters
        ----------
        ref_disp: float
            Reference displacement for the pushover analysis (e.g., yield displacement, or a baseline displacement).

        mu_levels: list
            Target ductility factors, which is used to scale the displacement at each cycle.

        dispIncr: int
            The number of displacement increments for each loading cycle.

        push_dir: int
            Direction of the pushover analysis.
            - 1 = X direction
            - 2 = Y direction
            - 3 = Z direction

        phi: list of floats
            The lateral load pattern shape. This is typically a mode shape or a predefined load distribution.
            For example, it can be the first-mode shape from the calibrateModel function.

        pflag: bool, optional, default=True
            If True, prints feedback during the analysis steps.

        num_steps: int, optional, default=200
            The number of steps for the cyclic pushover analysis.

        ansys_soe: string, optional, default='BandGeneral'
            System of equations solver to be used for the analysis.

        constraints_handler: string, optional, default='Transformation'
            The method used for handling constraint equations, such as enforcing displacement boundary conditions.

        numberer: string, optional, default='RCM'
            The numberer method used to assign equation numbers to degrees of freedom.

        test_type: string, optional, default='NormDispIncr'
            The type of test to be used for convergence in the solution of the linear system of equations.

        init_tol: float, optional, default=1e-5
            The initial tolerance for convergence.

        init_iter: int, optional, default=1000
            The maximum number of iterations for the solver to check convergence.

        algorithm_type: string, optional, default='KrylovNewton'
            The type of algorithm used to solve the system of equations (e.g., Krylov-Newtown method).

        Returns
        -------
        cpo_disps: numpy.ndarray
            An array containing the displacements at each floor at each step of the analysis.

        cpo_rxn: numpy.ndarray
            An array containing the base shear values, calculated as the sum of the reactions at the base.

        """

        # check ductility targets
        if mu_levels is None:
            mu_levels = [1, 2, 4, 6, 8, 10]

        # apply the load pattern
        ops.timeSeries("Linear", 1) # create timeSeries
        ops.pattern("Plain",1,1) # create a plain load pattern

        # define control nodes
        nodeList = ops.getNodeTags()
        control_node = nodeList[-1]
        pattern_nodes = nodeList[1:]
        rxn_nodes = [nodeList[0]]

        # quality control
        assert len(phi) == len(pattern_nodes), "phi length must match pattern_nodes"
        assert len(self.floor_masses) == len(pattern_nodes), "floor_masses length mismatch"

        # we can integrate modal patterns, inverse triangular, etc.
        for i in np.arange(len(pattern_nodes)):
            if push_dir == 1:
                if len(pattern_nodes)==1:
                    ops.load(pattern_nodes[i], 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                else:
                    ops.load(pattern_nodes[i], phi[i]*self.floor_masses[i], 0.0, 0.0, 0.0, 0.0, 0.0) ######### IT STARTS FROM ZERO

            elif push_dir == 2:
                if len(pattern_nodes)==1:
                    ops.load(pattern_nodes[i], 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
                else:
                    ops.load(pattern_nodes[i], 0.0, phi[i]*self.floor_masses[i], 0.0, 0.0, 0.0, 0.0)

            elif push_dir == 3:
                if len(pattern_nodes)==1:
                    ops.load(pattern_nodes[i], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
                else:
                    ops.load(pattern_nodes[i], 0.0, 0.0, phi[i]*self.floor_masses[i], 0.0, 0.0, 0.0)

        # Set up the initial objects
        ops.system(ansys_soe)
        ops.constraints(constraints_handler)
        ops.numberer(numberer)
        ops.test(test_type, init_tol, init_iter)
        ops.algorithm(algorithm_type)

        # Internally create push-pull cycle list with positive and negative displacements
        cycleDispList = []
        for mu in mu_levels:
            cycleDispList.append(ref_disp * mu)   # push positive
            cycleDispList.append(-ref_disp * mu)  # push negative
        dispNoMax = len(cycleDispList)


        # Give some feedback if requested
        if pflag:
            print(f"\n------ Cyclic Pushover with ductility levels: {mu_levels} ------")

        # Recording base shear
        cpo_rxn = [0.0]
        cpo_rxn = np.array(cpo_rxn)

        # Recording top displacement
        cpo_top_disp = [ops.nodeDisp(control_node, push_dir)]
        cpo_top_disp = np.array(cpo_top_disp)

        # Recording all displacements to estimate drifts
        cpo_disps = [[ops.nodeDisp(node, push_dir) for node in pattern_nodes]]
        cpo_disps = np.array(cpo_disps)

        # Initialize dissipated energy tracker
        energy_steps = [0.0]  # Initial energy is zero

        for d in range(dispNoMax):
            numIncr = dispIncr
            current_disp = ops.nodeDisp(control_node, push_dir)
            target_disp = cycleDispList[d]
            dU = (target_disp - current_disp) / numIncr
            ops.integrator('DisplacementControl', control_node, push_dir, dU)
            ops.analysis('Static')

            for l in range(numIncr):
                ok = ops.analyze(1)

                # If the analysis fails, try the following changes to achieve convergence
                if ok != 0:
                    print('FAILED: Trying relaxing convergence...')
                    ops.test(test_type, init_tol*0.01, init_iter)
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)
                if ok != 0:
                    print('FAILED: Trying relaxing convergence with more iterations...')
                    ops.test(test_type, init_tol*0.01, init_iter*10)
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)
                if ok != 0:
                    print('FAILED: Trying relaxing convergence with more iteration and Newton with initial then current...')
                    ops.test(test_type, init_tol*0.01, init_iter*10)
                    ops.algorithm('Newton', 'initialThenCurrent')
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)
                    ops.algorithm(algorithm_type)
                if ok != 0:
                    print('FAILED: Trying relaxing convergence with more iteration and Newton with initial...')
                    ops.test(test_type, init_tol*0.01, init_iter*10)
                    ops.algorithm('Newton', 'initial')
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)
                    ops.algorithm(algorithm_type)
                if ok != 0:
                    print('FAILED: Attempting a Hail Mary...')
                    ops.test('FixedNumIter', init_iter*10)
                    ok = ops.analyze(1)
                    ops.test(test_type, init_tol, init_iter)
                if ok != 0:
                    print('Analysis Failed')
                    break

            # Give some feedback if requested
            if pflag is True:
                curr_disp = ops.nodeDisp(control_node, push_dir)
                print('Currently pushed node ' + str(control_node) + ' to ' + str(curr_disp))

            # Get current displacement and base shear
            curr_disp = ops.nodeResponse(control_node, push_dir, 1)
            cpo_top_disp = np.append(cpo_top_disp, curr_disp)

            # Append displacement vector for all floors
            cpo_disps = np.append(cpo_disps, np.array([
                [ops.nodeResponse(node, push_dir, 1) for node in pattern_nodes]
            ]), axis=0)

            # Calculate current base shear
            ops.reactions()
            temp = 0
            for n in rxn_nodes:
                temp += ops.nodeReaction(n, push_dir)
            curr_rxn = -temp
            cpo_rxn = np.append(cpo_rxn, curr_rxn)

            # Calculate incremental energy (trapezoid rule)
            if len(cpo_top_disp) >= 2:
                dU = cpo_top_disp[-1] - cpo_top_disp[-2]
                avg_F = 0.5 * (cpo_rxn[-1] + cpo_rxn[-2])
                dE = abs(avg_F * dU)  # Energy in kN·m (assuming displacements in meters and force in kN)
                energy_steps.append(energy_steps[-1] + dE)

        pseudo_steps = np.arange(len(energy_steps))  # or use actual step counter if you prefer
        cpo_energy = np.column_stack((pseudo_steps, energy_steps))
        assert np.all(np.diff(cpo_energy[:,1]) >= 0), "Energy should be cumulative and increasing"
        
        ### Wipe the analysis objects
        ops.wipeAnalysis()

        return cpo_disps, cpo_rxn, cpo_energy

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
