# Nonlinear Time-History Analysis Demo

This Jupyter Notebook demonstrates an example application of nonlinear dynamic analysis on a multi-degree-of-freedom (MDOF) stick model where global and local response quantities such as peak storey drifts or peak floor accelerations are extracted.

The MDOF stick model is compiled in OpenSees using the `modeller` module, defining nodes, masses, elements, and storey-based force-deformation relationships in nonlinear springs (i.e., zero-length elements). Then, a single record is used to demonstrate the application of nonlinear time-history analyses using the `modeller` module.

Next, the `plotter` module is used to illustrate the seismic demand profiles (i.e., peak storey drift and peak floor acceleration quantities) along the height of the idealised MDOF stick-and-mass model.

NOTE: A demonstration of how to export response quantities from NLTHA is illustrated. This is extremely useful to provide OQ-VMTK's `postprocessor` and `plotter` modules with consistent and ready-to-implement input data formats for further data processing and handling. Therefore, users are encouraged to follow the same procedure. 
