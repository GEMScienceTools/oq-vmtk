Model Calibration Module
########################

.. module:: calibration
   :synopsis: A Python module for calibrating Multi-Degree-of-Freedom (MDOF) storey force-deformation relationships based on Single-Degree-of-Freedom (SDOF) capacity functions.

The ``calibration`` module provides a function ``calibrate_model`` that computes
MDOF storey forces, displacements, and mode shapes by transforming SDOF-based
capacity curves. The function accounts for factors such as the number of storeys,
building class, and the presence of soft-storey or frame structures. It applies
physical assumptions and simplifications, including uniform mass distribution and
standardized stiffness matrices.

Functions
---------

.. function:: calibrate_model(nst, gamma, sdof_capacity, isFrame, isSOS)

   Calibrates Multi-Degree-of-Freedom (MDOF) storey force-deformation relationships based on Single-Degree-of-Freedom (SDOF) capacity functions.

   :param nst: The number of storeys in the building (must be a positive integer).
   :type nst: int
   :param gamma: The SDOF-MDOF transformation factor. This factor adjusts the response of the MDOF system based on the SDOF capacity.
   :type gamma: float
   :param sdof_capacity: The SDOF spectral capacity data, where:
                         - Column 1 represents spectral displacements or accelerations.
                         - Column 2 represents spectral forces or accelerations.
                         - (For a trilinear/quadrilinear capacity curve) Additional columns may represent subsequent branches of the curve.
   :type sdof_capacity: array-like, shape (n, 2 or 3 or 4)
   :param isFrame: Flag indicating whether the building is a framed structure (True) or braced structure (False).
   :type isFrame: bool
   :param isSOS: Flag indicating whether the building contains a soft-storey (True) or not (False).
   :type isSOS: bool
   :return: MDOF floor masses, storey displacements, storey forces, and mode shape.
   :rtype: tuple(list of float, list of float, list of float, list of float)

   **Returns:**

   - **flm_mdof**: The MDOF floor masses, derived based on the mode shape and transformation factor.
   - **stD_mdof**: The MDOF storey displacements, adjusted for each floor and the applied SDOF capacity curve.
   - **stF_mdof**: The MDOF storey forces, computed based on the calibrated capacity functions.
   - **phi_mdof**: The expected mode shape for the MDOF system, normalized to have a unit norm.

   .. note::
      - If the building has a soft-storey, a modified stiffness matrix is used with reduced stiffness for the last floor.
      - The mode shape is derived using a generalized eigenvalue problem with mass and stiffness matrices.
      - The function handles various types of SDOF capacity curves (bilinear, trilinear, quadrilinear) to calibrate the MDOF system.
      - The effective mass for the SDOF system is computed assuming uniform mass distribution across floors.

References
----------
1) Lu X, McKenna F, Cheng Q, Xu Z, Zeng X, Mahin SA. An open-source framework for regional earthquake loss
   estimation using the city-scale nonlinear time history analysis. Earthquake Spectra. 2020;36(2):806-831.
   doi:10.1177/8755293019891724

2) Zhen Xu, Xinzheng Lu, Kincho H. Law, A computational framework for regional seismic simulation of buildings with
   multiple fidelity models, Advances in Engineering Software, Volume 99, 2016, Pages 100-110, ISSN 0965-9978,
   https://doi.org/10.1016/j.advengsoft.2016.05.014. (https://www.sciencedirect.com/science/article/pii/S0965997816301181)

3) EN 1998-1:2004 (Eurocode 8: Design of structures for earthquake resistance - Part 1: General rules, seismic actions,
   and rules for buildings)
