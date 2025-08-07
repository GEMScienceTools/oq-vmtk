# Pushover Analysis Demo

This Jupyter Notebook demonstrates an example application of monotonic static pushover and cyclic pushover analyses on a multi-degree-of-freedom (MDOF) stick model where global and local response quantities such as base shear, storey displacements and interstorey drifts are extracted.

The MDOF stick model is compiled in OpenSees using the `modeller` module, defining nodes, masses, elements, and storey-based force-deformation relationships in nonlinear springs (i.e., zero-length elements).

The storey-based force-deformation relationships are calibrated using a target capacity curve defined for an equivalent Single-Degree-of-Freedom (SDOF) system expressed in terms of spectral acceleration and displacement and using the `calibration` module of OQ-VMTK. The SDOF capacity is converted into an MDOF target by distributing strength and stiffness across the stories based on a modal shape, typically the first mode.
