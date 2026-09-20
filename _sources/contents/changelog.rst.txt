Change Log
##########

All notable changes to ``oq-vmtk`` are recorded here. The full machine-readable
changelog is maintained in
`CHANGELOG.md <https://github.com/GEMScienceTools/oq-vmtk/blob/main/CHANGELOG.md>`_
on GitHub.

v1.1.0 (in development)
------------------------

Added
~~~~~

- **Intensity Measure Selection module** (``openquake/vmtk/im_selection.py``):
  new ``imselection`` class implementing the Relative Sufficiency Measure (RSM)
  framework from Ebrahimian & Jalayer (2021). Supports Modified Cloud Analysis (MCA)
  and Incremental Dynamic Analysis (IDA) workflows. Methods: ``compute_efficiency_mca``,
  ``compute_efficiency_ida``, ``compute_proficiency_mca``, ``compute_proficiency_ida``,
  ``compute_rsm_mca``, ``compute_rsm_ida``, ``compute_rsm_general``, ``compare_ims``.

- **Demo notebook**: ``demos/IntensityMeasureSelection/IntensityMeasureSelection.ipynb``
  demonstrating MCA and IDA IM selection workflows.

- **``postprocessor`` — logistic collapse parameters**: ``cloud_dict['regression']``
  now exposes ``alpha0`` and ``alpha1`` (logistic intercept and slope) as scalar keys,
  enabling direct use in the RSM computation.

Changed
~~~~~~~

- **``postprocessor`` method renames** (no logic changes, backwards-incompatible):

  ============================================  ============================
  Old name                                      New name
  ============================================  ============================
  ``do_modified_cloud_analysis``                ``process_mca_results``
  ``do_multiple_stripe_analysis``               ``process_msa_results``
  ``do_incremental_dynamic_analysis``           ``process_ida_results``
  ============================================  ============================

  The old ``do_*`` names implied that the methods *run* the nonlinear analysis.
  They only postprocess already-computed structural response data; the new names
  reflect this clearly.

- **``imcalculator.get_duration_ims`` removed**: this convenience wrapper around
  ``get_arias_intensity``, ``get_cav``, and ``get_significant_duration`` has been
  deleted. Call each method individually.

- **Documentation**: all module pages now use ``.. autoclass::`` / ``.. automethod::``
  directives so the API reference updates automatically from source docstrings on
  every build. Pages are numbered (1–9) and each method appears as a numbered
  sub-section in the sidebar.

v1.0.0
------

Added or Changed
~~~~~~~~~~~~~~~~

- Stable source code for vulnerability-toolkit.
- Added AGPL v3 license.
- Added ``CONTRIBUTORS.txt``.
