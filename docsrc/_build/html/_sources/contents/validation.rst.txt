Validation and Quality Assurance
################################

``oq-vmtk`` follows the same testing philosophy as the OpenQuake engine:
**every module ships with unit tests whose expected values are obtained from an
independent implementation or a published reference**. The tests live in
`openquake/vmtk/tests/ <https://github.com/GEMScienceTools/oq-vmtk/tree/main/openquake/vmtk/tests>`_
and are executed on every pull request via the GitHub Actions matrix
(Linux, Windows, macOS arm64; Python 3.11, 3.12, 3.13).

Continuous Integration
======================

Three workflows guard the repository:

- ``linux_test.yml`` — installs the OpenQuake engine, the pinned requirements
  file, and ``oq-vmtk`` itself; runs the full ``pytest`` suite; rebuilds the
  Sphinx documentation and deploys to GitHub Pages.
- ``windows_test.yml`` — equivalent installation and test run on Windows.
- ``macos_arm_test.yml`` — equivalent installation and test run on macOS arm64.

A failing test on any platform blocks merging.

Reference-value Tests
=====================

Intensity Measure Calculator
----------------------------

The acceleration record bundled at
``openquake/vmtk/tests/test_data/acceleration.txt`` is used to verify every IM
returned by :class:`openquake.vmtk.imcalculator.imcalculator` against
independently computed reference values. The reference values are stored as
class-level attributes at the top of ``test_imcalculator.py``:

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Intensity measure
     - Reference value
     - Computed by
   * - PGA
     - 0.54557 g
     - ``imcalculator.get_amplitude_ims``
   * - PGV
     - 0.42661 m/s
     - ``imcalculator.get_amplitude_ims``
   * - PGD
     - 0.03304 m
     - ``imcalculator.get_amplitude_ims``
   * - :math:`Sa(T=0.3\,\mathrm{s})`
     - 1.30976 g
     - ``imcalculator.get_sa``
   * - :math:`Sa(T=0.6\,\mathrm{s})`
     - 0.78053 g
     - ``imcalculator.get_sa``
   * - :math:`Sa(T=1.0\,\mathrm{s})`
     - 0.31042 g
     - ``imcalculator.get_sa``
   * - :math:`AvgSa(T=0.3\,\mathrm{s})`
     - 1.20747 g
     - ``imcalculator.get_saavg``
   * - :math:`AvgSa(T=0.6\,\mathrm{s})`
     - 0.81096 g
     - ``imcalculator.get_saavg``
   * - :math:`AvgSa(T=1.0\,\mathrm{s})`
     - 0.43578 g
     - ``imcalculator.get_saavg``
   * - User-defined :math:`AvgSa` (T = 0.1–1.0 s)
     - 0.76748 g
     - ``imcalculator.get_saavg_user_defined``
   * - Arias Intensity
     - 1.99202 m/s
     - ``imcalculator.get_arias_intensity``
   * - CAV
     - 10.03464 m/s
     - ``imcalculator.get_cav``
   * - :math:`t_{5\!-\!95}` significant duration
     - 7.695 s
     - ``imcalculator.get_significant_duration``
   * - FIV3 (T = 0.3 s, α = 1.0, β = 0.7)
     - 0.07390 g
     - ``imcalculator.get_FIV3``

Each value is asserted to four decimal places (three for AI, CAV, and
significant duration) inside ``test_imcalculator.py``. Any drift in the
underlying numerics is caught immediately by CI.

Storey Loss Function Generator
------------------------------

The SLF generator is validated using the FEMA P-58 component inventory shipped
under ``demos/StoreyLossFunctionGeneration/in/`` and a reduced fixture in
``openquake/vmtk/tests/test_data/slf_inventory.csv``. The unit tests cover:

- input validation (missing columns, wrong EDP type, invalid component IDs);
- reproducibility of Monte-Carlo realisations under a fixed random seed;
- shape and monotonicity of the regressed SLFs across the four supported
  functional forms (Weibull, Papadopoulos, Generalised Pareto, Lognormal).

The SLF implementation itself is benchmarked against the original
Python-based generator described by Shahnazaryan, O'Reilly and Monteiro
(`COMPDYN 2021 <https://doi.org/10.7712/120121.8659.18567>`_,
`Earthquake Spectra 2021 <https://doi.org/10.1177/87552930211023523>`_).

Modeller, Calibration, Postprocessor, Plotter
---------------------------------------------

Each module has a dedicated ``test_<module>.py`` file in
``openquake/vmtk/tests/``. The modeller and calibration tests verify the
period and mode shapes produced for canonical SDOF and MDOF reference
configurations against textbook closed-form solutions. The postprocessor tests
verify the lognormal, GLM, and ordinal fragility fits against synthetic
datasets with analytically known parameters.

Reproducing Results Locally
===========================

From the project root:

.. code-block:: bash

   pytest -v openquake/vmtk/tests/

Runtime is approximately 1–2 minutes on a modern laptop. To run a single
module's tests:

.. code-block:: bash

   pytest -v openquake/vmtk/tests/test_imcalculator.py

Contributing New Tests
======================

When adding a new feature, contributors are expected to ship at least one unit
test whose expected values are computed independently — either from a
published reference, an independent code, or an analytical solution. Pull
requests that add new functions without such tests will be requested to add
them before merge. See
`contribute_guidelines.md <https://github.com/GEMScienceTools/oq-vmtk/blob/main/contribute_guidelines.md>`_
for the full contribution policy.
