Change Log
##########

All notable changes to ``oq-vmtk`` are recorded here. The full machine-readable
changelog is maintained in
`CHANGELOG.md <https://github.com/GEMScienceTools/oq-vmtk/blob/main/CHANGELOG.md>`_
on GitHub.

v1.2.0 (2026-10-09)
-------------------

First stable release of OQ-VMTK. This version is the starting point of the
changelog; the API documented here is the one supported going forward.

Modules
~~~~~~~

- ``calibration``: ``calibrate_model()`` derives storey-level force–deformation
  properties of MDOF stick-and-mass models from SDOF capacity curves (assumed
  power-law or eigenvector first-mode shape; soft-storey option).
- ``modeller``: compiles and runs SDOF/MDOF models in OpenSeesPy — gravity and
  modal analysis, static (SPO) and cyclic (CPO) pushover, nonlinear
  time-history analysis (single records and sequences) and incremental dynamic
  analysis (IDA), with collapse detection and animated outputs.
- ``imcalculator``: intensity measures (PGA, PGV, PGD, SA, AvgSA, Arias
  Intensity, CAV, D5–95, FIV3) and RotDxx spectra; PGV/PGD are computed from
  drift-corrected (Boore, 2005) histories. RotDxx (e.g. RotD50) versions of
  PGA/PGV/PGD, CAV, Arias Intensity, D5–95, AvgSA and FIV3 are computed over
  180 rotation angles (``get_rotdxx_*``).
- ``imselection``: IM ranking by efficiency, proficiency, practicality and
  relative sufficiency measure (RSM).
- ``postprocessor``: probabilistic seismic demand models and fragility
  functions from Modified Cloud Analysis (classical, bootstrap, MCMC), Multiple
  Stripe Analysis and IDA; vulnerability functions with explicit uncertainty
  propagation; ``calculate_risk`` for AADP/AALR.
- ``slfgenerator``: Monte Carlo storey loss functions from component
  inventories (independent or correlated components), returning the empirical
  16th/50th/84th percentiles of the loss ratio vs. EDP.
- ``plotter``: figures for the whole workflow with a uniform plotting grid.
- ``utilities``: I/O helpers and OpenQuake Engine interoperability.

Demos and documentation
~~~~~~~~~~~~~~~~~~~~~~~

- Thirteen demo notebooks covering the full workflow, including the
  ``EQSpectraExample`` supplement to the EQ Spectra paper.
- Sphinx documentation, ``README.md`` and ``CITATION.cff``.
- Python 3.11–3.13 supported; CI on Linux, Windows and macOS ARM64 with
  platform-specific pinned requirements files.
