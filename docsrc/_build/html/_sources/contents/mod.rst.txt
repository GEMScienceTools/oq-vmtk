Model Building and Analysis
###########################

The ``modeller`` class builds and analyses stick-and-mass MDOF structural models
using the OpenSees framework. Each storey is represented by a Pinching4 hysteretic
spring. Supported analysis types include gravity analysis, modal analysis, static
pushover (SPO), and nonlinear time-history analysis (NRHA).

.. toctree::

   mod/init
   mod/compile_model
   mod/plot_model
   mod/do_gravity_analysis
   mod/do_modal_analysis
   mod/do_spo_analysis
   mod/do_nrha_analysis

References
----------

1. Minjie, Zhu. McKenna, F. and Scott, M.H. (2018). "OpenSeesPy: Python library for the OpenSees finite element
   framework", *SoftwareX*, Volume 7, 2018, Pages 6-11, ISSN 2352-7110,
   https://doi.org/10.1016/j.softx.2017.10.009.
