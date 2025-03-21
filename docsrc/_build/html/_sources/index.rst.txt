.. mbt documentation master file, created by
   sphinx-quickstart on Thu Jan 24 16:06:36 2019.
      You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the OpenQuake Vulnerability Modellers Toolkit's documentation!
#########################################################################

The OpenQuake Vulnerability Modellers Toolkit (*oq-vmtk*) is an open source
library that provides an OpenSeesPy-based environment for modelling idealised
building class models such as single (SDOF) and multi-degree-of-freedom
(MDOF) systems and to carry out analysis via linear, nonlinear static and nonlinear
dynamic  approaches for regional vulnerability and risk assessment applications.
The vulnerability toolkit is developed by the Risk Team at the Global Earthquake
Model (GEM) Foundation. Contribution from external users are very welcome!

*oq-vmtk* code is hosted on github at the following link
https://github.com/GEMScienceTools/oq-vmtk.

Currently the oq-vmtk includes eight sub-modules:

* *Intensity Measure Calculator (im_calculator)* contains code used for
processing spectra and intensity measure types from ground-motion records;
* *Model Calibration (calibration)* contains code used for calibrating MDOF models
based on SDOF low-level parameters;
* *Model Building (modeller)* contains code used to compile SDOF and MDOF models
in OpenSeesPy and run distinct analysis such as linear (i.e., modal analysis),
nonlinear static (e.g.,static and cyclic pushovers) and nonlinear time-history
analyses;
* *Fragility and Vulnerability Analysis (postprocessor)* contains code for
postprocessing cloud and multiple stripe analyses to derive fragility and
vulnerability models;
* *Storey Loss Function Generator (slf_generator)* contains code for generating
storey loss functions for a more refined loss assessment of building components;
* *Model Plotting (plotter)* contains code for visualising and graphically
interpreting oq-vmtk outputs;
* *Units and Utilities* contains miscellaneous code used for assigning units of
measurements to OpenSeesPy models and to carry out other file management and
variable handling tasks, respectively.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   contents/installation
   contents/imc
   contents/cal
   contents/mod
   contents/pos
   contents/slf
   contents/plo


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
