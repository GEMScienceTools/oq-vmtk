Model Calibration
#################

The ``calibration`` class transforms Single-Degree-of-Freedom (SDOF) spectral
capacity parameters into Multi-Degree-of-Freedom (MDOF) storey force-deformation
relationships. It handles building classification, mass and stiffness matrix
assembly, eigenvalue-based modal analysis, period scaling, storey force-drift
distribution, and optional OpenSees static pushover verification.

.. toctree::

   cal/init
   cal/calibrate_model

References
----------

1. Lu X, McKenna F, Cheng Q, Xu Z, Zeng X, Mahin SA. An open-source framework for regional earthquake loss
   estimation using the city-scale nonlinear time history analysis. Earthquake Spectra. 2020;36(2):806-831.
   doi:10.1177/8755293019891724

2. Zhen Xu, Xinzheng Lu, Kincho H. Law, A computational framework for regional seismic simulation of buildings
   with multiple fidelity models, Advances in Engineering Software, Volume 99, 2016, Pages 100-110,
   https://doi.org/10.1016/j.advengsoft.2016.05.014.

3. EN 1998-1:2004 (Eurocode 8: Design of structures for earthquake resistance - Part 1: General rules, seismic
   actions, and rules for buildings)
