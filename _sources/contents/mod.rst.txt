Model Building Module
#####################

.. module:: modeller
   :synopsis: A Python module for modeling and analyzing multi-degree-of-freedom (MDOF) oscillators using OpenSees.

The ``modeller`` module provides a class ``modeller`` that allows users to create,
analyze, and visualize structural models for dynamic and static analyses.
The module supports various types of analyses, including gravity analysis, modal
analysis, static pushover analysis, cyclic pushover analysis, and nonlinear
time-history analysis. The module is built on top of the OpenSees framework,
enabling advanced structural analysis capabilities.

Classes
-------

.. class:: modeller(number_storeys, floor_heights, floor_masses, storey_disps, storey_forces, degradation)

   A class to model and analyze multi-degree-of-freedom (MDOF) oscillators using OpenSees.

   :param number_storeys: The number of storeys in the building model.
   :type number_storeys: int
   :param floor_heights: List of floor heights in meters.
   :type floor_heights: list
   :param floor_masses: List of floor masses in tonnes.
   :type floor_masses: list
   :param storey_disps: Array of storey displacements (size = number of storeys, CapPoints).
   :type storey_disps: np.array
   :param storey_forces: Array of storey forces (size = number of storeys, CapPoints).
   :type storey_forces: np.array
   :param degradation: Boolean to enable or disable hysteresis degradation.
   :type degradation: bool

   .. method:: create_Pinching4_material(mat1Tag, mat2Tag, storey_forces, storey_disps, degradation)

      Creates a Pinching4 material model for the MDOF oscillator.

      :param mat1Tag: Material tag for the first material in the Pinching4 model.
      :type mat1Tag: int
      :param mat2Tag: Material tag for the second material in the Pinching4 model.
      :type mat2Tag: int
      :param storey_forces: Array of storey forces at each storey in the model.
      :type storey_forces: np.array
      :param storey_disps: Array of storey displacements corresponding to the forces.
      :type storey_disps: np.array
      :param degradation: Boolean flag to enable or disable hysteresis degradation.
      :type degradation: bool
      :return: None
      :rtype: None

   .. method:: compile_model()

      Compiles and sets up the MDOF oscillator model in OpenSees.

      :return: None
      :rtype: None

   .. method:: plot_model(display_info=True)

      Plots the 3D visualization of the OpenSees model.

      :param display_info: If True, displays additional information (coordinates and node ID) next to each node in the plot.
      :type display_info: bool, optional
      :return: None
      :rtype: None

   .. method:: do_gravity_analysis(nG=100, ansys_soe='UmfPack', constraints_handler='Transformation', numberer='RCM', test_type='NormDispIncr', init_tol=1.0e-6, init_iter=500, algorithm_type='Newton', integrator='LoadControl', analysis='Static')

      Performs gravity analysis on the MDOF system.

      :param nG: Number of gravity analysis steps to perform.
      :type nG: int, optional
      :param ansys_soe: The system of equations type to be used in the analysis.
      :type ansys_soe: string, optional
      :param constraints_handler: The constraints handler determines how the constraint equations are enforced.
      :type constraints_handler: string, optional
      :param numberer: The degree-of-freedom numberer defines how DOFs are numbered.
      :type numberer: string, optional
      :param test_type: Defines the test type used to check the convergence of the solution.
      :type test_type: string, optional
      :param init_tol: The tolerance criterion for checking convergence.
      :type init_tol: float, optional
      :param init_iter: The maximum number of iterations to check for convergence.
      :type init_iter: int, optional
      :param algorithm_type: Defines the solution algorithm used in the analysis.
      :type algorithm_type: string, optional
      :param integrator: Defines the integrator for the analysis.
      :type integrator: string, optional
      :param analysis: Defines the type of analysis to be performed.
      :type analysis: string, optional
      :return: None
      :rtype: None

   .. method:: do_modal_analysis(num_modes=3, solver='-genBandArpack', doRayleigh=False, pflag=False)

      Performs modal analysis to determine natural frequencies and mode shapes.

      :param num_modes: The number of modes to consider in the analysis.
      :type num_modes: int, optional
      :param solver: The type of solver to use for the eigenvalue problem.
      :type solver: string, optional
      :param doRayleigh: Flag to enable or disable Rayleigh damping in the modal analysis.
      :type doRayleigh: bool, optional
      :param pflag: Flag to control whether to print the modal analysis report.
      :type pflag: bool, optional
      :return: Periods of vibration and mode shapes.
      :rtype: tuple(np.array, list)

   .. method:: do_spo_analysis(ref_disp, disp_scale_factor, push_dir, phi, pflag=True, num_steps=200, ansys_soe='BandGeneral', constraints_handler='Transformation', numberer='RCM', test_type='EnergyIncr', init_tol=1.0e-5, init_iter=1000, algorithm_type='KrylovNewton')

      Performs static pushover analysis (SPO) on the MDOF system.

      :param ref_disp: The reference displacement at which the analysis starts.
      :type ref_disp: float
      :param disp_scale_factor: The scale factor applied to the reference displacement.
      :type disp_scale_factor: float
      :param push_dir: The direction in which the pushover load is applied.
      :type push_dir: int
      :param phi: The lateral load pattern shape.
      :type phi: list of floats
      :param pflag: Flag to print (or not) the pushover analysis steps.
      :type pflag: bool, optional
      :param num_steps: The number of steps to increment the pushover load.
      :type num_steps: int, optional
      :param ansys_soe: The type of system of equations solver to use.
      :type ansys_soe: string, optional
      :param constraints_handler: The constraints handler object to determine how constraint equations are enforced.
      :type constraints_handler: string, optional
      :param numberer: The degree-of-freedom (DOF) numberer object.
      :type numberer: string, optional
      :param test_type: The type of test to use for the linear system of equations.
      :type test_type: string, optional
      :param init_tol: The tolerance criterion to check for convergence.
      :type init_tol: float, optional
      :param init_iter: The maximum number of iterations to perform when checking for convergence.
      :type init_iter: int, optional
      :param algorithm_type: The type of algorithm used to solve the system.
      :type algorithm_type: string, optional
      :return: Displacements, base shear, and spring forces.
      :rtype: tuple(np.array, np.array, np.array, np.array)

   .. method:: do_cpo_analysis(ref_disp, mu, numCycles, push_dir, dispIncr, pflag=True, num_steps=200, ansys_soe='BandGeneral', constraints_handler='Transformation', numberer='RCM', test_type='NormDispIncr', init_tol=1.0e-5, init_iter=1000, algorithm_type='KrylovNewton')

      Performs cyclic pushover analysis (CPO) on the MDOF system.

      :param ref_disp: Reference displacement for the pushover analysis.
      :type ref_disp: float
      :param mu: Target ductility factor.
      :type mu: float
      :param numCycles: The number of displacement cycles to be performed.
      :type numCycles: int
      :param push_dir: Direction of the pushover analysis.
      :type push_dir: int
      :param dispIncr: The number of displacement increments for each loading cycle.
      :type dispIncr: float
      :param pflag: Flag to print feedback during the analysis steps.
      :type pflag: bool, optional
      :param num_steps: The number of steps for the cyclic pushover analysis.
      :type num_steps: int, optional
      :param ansys_soe: System of equations solver to be used for the analysis.
      :type ansys_soe: string, optional
      :param constraints_handler: The method used for handling constraint equations.
      :type constraints_handler: string, optional
      :param numberer: The numberer method used to assign equation numbers to degrees of freedom.
      :type numberer: string, optional
      :param test_type: The type of test to be used for convergence in the solution of the linear system of equations.
      :type test_type: string, optional
      :param init_tol: The initial tolerance for convergence.
      :type init_tol: float, optional
      :param init_iter: The maximum number of iterations for the solver to check convergence.
      :type init_iter: int, optional
      :param algorithm_type: The type of algorithm used to solve the system of equations.
      :type algorithm_type: string, optional
      :return: Displacements and base shear.
      :rtype: tuple(np.array, np.array)

   .. method:: do_nrha_analysis(fnames, dt_gm, sf, t_max, dt_ansys, nrha_outdir, pflag=True, xi=0.05, ansys_soe='BandGeneral', constraints_handler='Plain', numberer='RCM', test_type='NormDispIncr', init_tol=1.0e-6, init_iter=50, algorithm_type='Newton')

      Performs nonlinear time-history analysis (NRHA) on the MDOF system.

      :param fnames: List of file paths to the ground motion records for each direction (X, Y, Z).
      :type fnames: list
      :param dt_gm: Time-step of the ground motion records.
      :type dt_gm: float
      :param sf: Scale factor to apply to the ground motion records.
      :type sf: float
      :param t_max: The maximum time duration for the analysis.
      :type t_max: float
      :param dt_ansys: The time-step at which the analysis will be conducted.
      :type dt_ansys: float
      :param nrha_outdir: Directory where temporary output files are saved during the analysis.
      :type nrha_outdir: string
      :param pflag: Flag to print progress updates during the analysis.
      :type pflag: bool, optional
      :param xi: The inherent damping ratio used in the analysis.
      :type xi: float, optional
      :param ansys_soe: Type of the system of equations solver to be used in the analysis.
      :type ansys_soe: string, optional
      :param constraints_handler: The method used to handle constraints in the analysis.
      :type constraints_handler: string, optional
      :param numberer: The numberer object determines the equation numbering used in the analysis.
      :type numberer: string, optional
      :param test_type: Type of convergence test used during the analysis.
      :type test_type: string, optional
      :param init_tol: Initial tolerance for the convergence test.
      :type init_tol: float, optional
      :param init_iter: Maximum number of iterations allowed during each time step for the analysis to converge.
      :type init_iter: int, optional
      :param algorithm_type: Type of algorithm used to solve the system of equations.
      :type algorithm_type: string, optional
      :return: Control nodes, convergence status, peak drifts, peak accelerations, and peak displacements.
      :rtype: tuple(list, int, np.array, np.array, float, string, int, float, string, int, np.array)

References
----------
1) Minjie, Zhu. McKenna, F. and Scott, M.H. (2018). "OpenSeesPy: Python library for the OpenSees finite element framework", *SoftwareX*, Volume 7, 2018, Pages 6-11, ISSN 2352-7110, https://doi.org/10.1016/j.softx.2017.10.009.
