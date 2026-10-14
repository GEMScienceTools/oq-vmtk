# Changelog

## Unreleased

### Changed
- `postprocessor.calculate_risk` now distinguishes rates from probabilities.
  The integral of a fragility curve against the hazard curve is an average
  annual damage *rate*; it is converted to the Average Annual Damage
  Probability assuming a Poisson process, `P = 1 - exp(-rate)` (default;
  `as_probability=False` returns the rate). A new required keyword-only
  argument `curve_type` (`'fragility'` or `'vulnerability'`) states what the
  input curve is; vulnerability curves (AALR) are never converted. Existing
  calls must add `curve_type=...`.
- `plotter.plot_demand_profiles`: new `drift_xlim` and `accel_xlim` arguments.
- `plotter.plot_vulnerability_function`: fewer x-axis tick labels.
- `slfgenerator` follows FEMA P-58 and returns the expected loss. Results
  differ from v1.2.0; there is no switch back to the previous behaviour.
  - Fragility columns are the lognormal median and dispersion (`DSk, Median`,
    `DSk, Total Dispersion`); they were read as a mean and coefficient of
    variation.
  - Every component has its own capacity, held over the EDP range; one random
    number per realisation and EDP was shared by all components, which made
    them perfectly correlated and left the correlation tree without effect.
  - Unit costs follow the FEMA P-58 conventions: `Lognormal` = median and
    dispersion (was mean and CoV), `Normal` = mean and CoV, truncated at zero.
    Costs are no longer truncated to integers.
  - New optional inventory columns: `DS Hierarchy` (e.g.
    `Seq(DS1,MutEx(DS2,DS3))`, `Simul(DS1,DS2,DS3,DS4)`) and
    `DSk, Probability` for mutually exclusive and simultaneous damage states,
    sampled explicitly; `Correlated` (FEMA P-58 flag; `NO` damages every unit
    independently, default `YES`); `Replacement Cost` per unit.
  - Correlation trees act on limit states, are applied unit by unit, may be
    chained, and are fully validated (cycles, forcing from DS0, decreasing or
    out-of-range forced states raise `ValueError`).
  - The storey loss ratio is bounded: each realisation is divided by the
    replacement cost of the group's components at the same unit costs. A
    component's unit costs share one price level per realisation (repair and
    replacement of the same item), and no unit is repaired for more than its
    replacement cost, so the ratio lies in [0, 1] and is 1 when everything is
    replaced. Its mean and variance can therefore always be represented by a
    Beta distribution.
  - `generate()` output: `slf` is now the mean loss ratio (was the Monte
    Carlo median), with new `slf_std`, `slf_cov`, `slf_50th` and
    `replacement_cost`. `replacement_cost` defaults to `None` (the bounded
    ratio above); a number divides every realisation by that fixed cost.
    The cache adds `ratio_of_expectations` (exact expected repair cost over
    expected replacement cost, a check on `slf`) and `replacement_value`.
  - New `seed` argument; each component draws from its own random stream,
    so models that differ only by their correlation tree share their random
    numbers. `realizations` defaults to 1000 (was 20).
    `edp_range` may start at 0 and is no longer modified (its first value was
    overwritten with 1e-20).
  - New methods `limit_state_probabilities`, `expected_component_loss`
    (exact),
    `apply_correlation_tree`; `fragility_function` returns only the fragility
    curves (keyed `LSk`). `do_monte_carlo_simulations`,
    `validate_ds_dependence` and `calculate_costs` are removed. The cache holds
    `(realizations, n_edp)` arrays instead of per-realisation dicts.
  - `grouping_flag=False` works (it raised `AttributeError`), and importing
    the module no longer silences all Python warnings.
  - Demo inventories gain examples of each damage state hierarchy; the
    elevators are simultaneous, as in FEMA P-58.
- `plotter.plot_slf_model` draws the expected loss (`slf`) and the median
  (`slf_50th`), and a sample of `max_realizations` realisations (default 200)
  as points.

## v1.2.0 — 2026-10-09

First stable release of OQ-VMTK. This version is the starting point of the
changelog; the API documented here is the one supported going forward.

### Modules
- `calibration`: `calibrate_model()` derives storey-level force–deformation
  properties of MDOF stick-and-mass models from SDOF capacity curves
  (assumed power-law or eigenvector first-mode shape; soft-storey option).
- `modeller`: compiles and runs SDOF/MDOF models in OpenSeesPy — gravity and
  modal analysis, static (SPO) and cyclic (CPO) pushover, nonlinear
  time-history analysis (single records, sequences) and incremental dynamic
  analysis (IDA), with collapse detection and animated outputs.
- `imcalculator`: intensity measures (PGA, PGV, PGD, SA, AvgSA, Arias
  Intensity, CAV, D5–95, FIV3), RotDxx spectra; PGV/PGD are computed from
  drift-corrected (Boore, 2005) histories. RotDxx (e.g. RotD50) versions of
  PGA/PGV/PGD, CAV, Arias Intensity, D5–95, AvgSA and FIV3 are computed over
  180 rotation angles (`get_rotdxx_*`).
- `imselection`: IM ranking by efficiency, proficiency, practicality and
  relative sufficiency measure (RSM).
- `postprocessor`: probabilistic seismic demand models and fragility
  functions from Modified Cloud Analysis (classical, bootstrap, MCMC),
  Multiple Stripe Analysis and IDA; vulnerability functions with explicit
  uncertainty propagation; `calculate_risk` for AADP/AALR.
- `slfgenerator`: Monte Carlo storey loss functions from component
  inventories (independent or correlated components), returning the
  empirical 16th/50th/84th percentiles of the loss ratio vs. EDP.
- `plotter`: figures for the whole workflow with a uniform plotting grid.
- `utilities`: I/O helpers and OpenQuake Engine interoperability.

### Demos and documentation
- Thirteen demo notebooks covering the full workflow, including the
  `EQSpectraExample` supplement to the EQ Spectra paper.
- Sphinx documentation, `README.md` and `CITATION.cff`.
- Python 3.11–3.13 supported; CI on Linux, Windows and macOS ARM64 with
  platform-specific pinned requirements files.
