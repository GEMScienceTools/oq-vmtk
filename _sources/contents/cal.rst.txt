Model Calibration
#################

The ``calibrate_model`` function transforms Single-Degree-of-Freedom (SDOF)
spectral capacity parameters into Multi-Degree-of-Freedom (MDOF) storey
force-deformation relationships. The first-mode shape is either an assumed
power law (frame buildings up to 12 storeys) or the first eigenvector of a
uniform tri-diagonal stiffness matrix (everything else), softened at the
ground floor for soft-storey buildings. Only the first mode is used; no
period-matching or OpenSees verification is performed.

.. toctree::

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
