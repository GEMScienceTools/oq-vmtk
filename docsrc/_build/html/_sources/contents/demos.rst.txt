Demos
#####

The repository ships with a set of Jupyter notebooks under the
`demos/ <https://github.com/GEMScienceTools/oq-vmtk/tree/main/demos>`_
folder that walk through the typical ``oq-vmtk`` workflow end-to-end. Each
notebook is self-contained: it lists the modules it uses, loads its own input
files, and saves outputs to its own ``out/`` directory.

Launching JupyterLab
====================

JupyterLab is pulled in as a transitive dependency when ``oq-vmtk`` is
installed, so you do not need to install it separately. From the project root
with the virtual environment activated:

.. code-block:: bash

   # On Windows
   .venv\Scripts\activate
   # On Linux/macOS
   source .venv/bin/activate

   jupyter-lab

Then navigate to the ``demos/`` directory in the JupyterLab file browser and
open the notebook of interest.

Available Demos
===============

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Notebook
     - What it shows
   * - ``IntensityMeasureProcessing``
     - Compute response spectra and a wide range of intensity measures (PGA,
       PGV, PGD, Sa, AvgSa, Arias intensity, CAV, significant duration, FIV3,
       RotDxx) from raw acceleration records using the ``imcalculator``.
   * - ``IntensityMeasureSelection``
     - Compare candidate intensity measures using information-theoretic
       sufficiency metrics from the ``imselection`` module.
   * - ``ModelCompilation``
     - Compile single- and multi-degree-of-freedom OpenSeesPy models from
       low-level parameters; demonstrates the ``modeller`` and ``calibration``
       modules.
   * - ``ModalAnalysis``
     - Run modal analysis on a compiled MDOF model and extract periods and
       mode shapes.
   * - ``PushoverAnalysis``
     - Static and cyclic pushover on an MDOF stick model — base shear,
       interstorey drift, and energy dissipation.
   * - ``NonlinearTimeHistoryAnalysis``
     - Nonlinear response-history analysis using ground-motion records, with
       postprocessing of peak storey drifts and floor accelerations.
   * - ``CloudAnalysis``
     - End-to-end cloud-analysis workflow producing fragility and
       vulnerability functions from MDOF response.
   * - ``MultipleStripeAnalysis``
     - Multiple-stripe analysis variant of the same end-to-end workflow.
   * - ``IncrementalDynamicAnalysis``
     - Incremental dynamic analysis (IDA) with collapse-fragility derivation.
   * - ``FragilityAnalysis``
     - Multiple fragility-fitting approaches (lognormal CDF, GLM, ordinal),
       fragility rotation, and additional epistemic uncertainty.
   * - ``StoreyLossFunctionGeneration``
     - Generate Storey Loss Functions from a component inventory using the
       ``slfgenerator`` module. Bundled inventories live in
       ``demos/StoreyLossFunctionGeneration/in/``.
   * - ``StoreyLossFunctionApplication``
     - Apply previously generated SLFs to derive nonstructural-component
       vulnerability functions.

Each demo folder contains a short ``README.md`` describing the inputs,
outputs, and the modules being exercised.

Suggested Reading Order
=======================

For first-time users, we recommend the following order:

1. ``IntensityMeasureProcessing`` — get familiar with the IM types.
2. ``ModelCompilation`` — build an MDOF stick model.
3. ``NonlinearTimeHistoryAnalysis`` — run dynamic analysis.
4. ``CloudAnalysis`` — full vulnerability workflow.
5. ``StoreyLossFunctionGeneration`` and ``StoreyLossFunctionApplication`` —
   add component-level loss modelling.
