# Changelog

## [Unreleased]

### Added
- `plotter.plot_vulnerability_function`: new optional `xlims` parameter (`[min, max]`) to restrict the plotted intensity range. Data arrays are filtered before plot creation so the violin axis is never distorted.

### Changed
- `plotter.plot_modes`: mode titles are now placed via `fig.text()` centred over both the X- and Y-displacement panels instead of being attached only to the left panel. Font sizes for node annotations, axis labels, tick marks, and panel titles, and the vertical spacing parameters (`hspace`, `top`), now scale adaptively with the number of grid rows and nodes per panel, preventing text and title overlap when plotting many modes.
- `plotter.plot_vulnerability_function`: x-axis tick label font size reduced to prevent overlap when many intensity levels are present.
- Demo `NonlinearTimeHistoryAnalysis`: introduction and section descriptions updated to match the concise style used across other demos; verbose bullet-point input/output argument lists removed.
- Demo `PushoverAnalysis`: cyclic pushover protocol updated to a two-repetition-per-level scheme (ATC-24 / FEMA 461 style), producing 20 full hysteresis loops instead of 11.
- Demo `StoreyLossFunctionApplication`: loss vs. IM fitting replaced with a power-law regression (OLS in log space, monotone by construction) paired with a Beta distribution dispersion model; COV estimated per quantile bin from the cloud scatter and propagated through the collapse-conditioning step using the law of total variance; summary tables of IML / mean loss / COV printed at both the non-conditioned and conditioned stages.
- `README.md`: all module and demo descriptions updated; badge URLs corrected (`.svg` suffix removed from shields.io GitHub badges; Zenodo DOI badge migrated to shields.io static endpoint); contributor shield URLs fixed; installation, license, citation, and references sections added.
- Demo `README.md` files: titles and descriptions corrected across `MultipleStripeAnalysis`, `FragilityAnalysis`, `IntensityMeasureProcessing`; `IntensityMeasureSelection/README.md` created; `demos/README.md` index updated to list all 13 demos.

### Fixed
- `plotter.plot_fragility_from_mca`: `title` parameter was silently ignored (variable was set but `ax.set_title()` was never called); title is now correctly applied to the figure.
- `plotter.plot_vulnerability_function`: Beta distribution sampling returned all-NaN when CoV was zero (division by zero produced `inf` parameters); now returns a point mass at the mean for the degenerate case.

### Renamed
- `imcalculator.get_velocity_displacement_history` → `imcalculator.get_vel_disp_history`. Documentation and examples updated accordingly.

---

## v1.1.0 — 2026-05-12

### Added
- **Incremental Dynamic Analysis (IDA)**: New `do_ida_analysis()` method in `modeller.py` runs nonlinear response history analyses at progressively scaled ground-motion intensities using a hunt-and-fill procedure (truncated and non-truncated). Results are post-processed by `process_ida_results()` in `postprocessor.py`, producing fragility functions and vulnerability curves via logistic regression and lognormal fitting. IDA plots (stripe curves, fragility, vulnerability) added to `plotter.py`.
- **IDA demo**: New `IncrementalDynamicAnalysis` notebook with the FEMA P695 far-field ground-motion record set (44 records).
- **MCMC for Modified Cloud Analysis**: Markov Chain Monte Carlo method added to `postprocessor.py` for MCA fragility derivation, alongside classical and bootstrap MCA plotter functions in `plotter.py`.
- **IM Efficiency and Sufficiency module** (`imselection.py`): New module implementing efficiency (dispersion-based IM ranking), practicality (regression slope), proficiency (κ metric), and the Relative Sufficiency Metric (RSM) for both MCA and IDA. Includes `compare_ims()` for tabulated multi-IM comparison.
- **RotDxx spectral calculations**: `get_rotdxx()` added to `imcalculator.py` to compute RotD50/RotD100 and arbitrary rotation-percentile response spectra from two horizontal components.
- **Structural analysis animations**: Animated deformed-shape outputs for SPO, CPO, and NRHA analyses via `modeller.py`. Animated mode shape visualisation via `plot_modes()` in `plotter.py`. All demo notebooks updated with animated GIF outputs.
- **COV calculations and DS threshold variability**: Added coefficient-of-variation methods and DS threshold variability as an input argument for NLTHA post-processing methods in `postprocessor.py`.
- **macOS ARM64 CI workflow** (`macos_arm_test.yml`) and platform-specific requirements files for Linux, Windows, and macOS ARM64.
- **Python 3.13 support** in `pyproject.toml` and CI workflows.
- `CITATION.cff` added.
- `README.md` files added for IDA, MSA, MCA, ModalAnalysis, and ModelCompilation demos.

### Changed
- **Calibration methodology**: `calibration.py` updated to a displacement-based design methodology.
- **Variable renames**: `pflag` → `pFlag` and `floor_heights` → `storey_heights` across `modeller.py`, `calibration.py`, and all dependents.
- **Class rename**: `IMCalculator` renamed to `imcalculator`; module `slf_generator.py` renamed to `slfgenerator.py`.
- **AAL and AADP refactor**: `calculate_average_annual_loss()` and `calculate_average_annual_damage_probability()` restructured in `postprocessor.py`.
- All core modules (`modeller.py`, `calibration.py`, `postprocessor.py`, `plotter.py`, `imcalculator.py`, `slfgenerator.py`, `units.py`, `utilities.py`) refactored to PEP8 standards.
- All demo notebooks and unit tests updated for PEP8 compliance.
- Input ground-motion records for demos relocated into `in/records/` subdirectories.
- Default response spectrum resolution increased to 500 points.
- CI upgraded to Node.js 20 actions; `GITHUB_TOKEN` added to workflows to prevent API rate-limiting.
- `scipy` pinned to `>=1.15.3`; `statsmodels` wheel added for macOS ARM64.

### Fixed
- `postprocessor.py`: Fixed `NoneType` export for non-lognormal fragility methods.
- `postprocessor.py`: Stabilised logistic regression when bootstrap produces too few collapses.
- `postprocessor.py`: Fixed out-of-bound beta values.
- `slfgenerator.py`: Fixed sampling bug.
- `plotter.py`: Fixed `RecursionError` in `_show()` (changed `self._show()` to `plt.show()`).
- `imcalculator.py`: Fixed multiple bugs in IM calculation routines.
- `modeller.py`: Fixed node displacement and acceleration storage allocation in `do_nrha_analysis()`.
- `modeller.py`: Fixed `openseespy` import to be OS-conditional.
- Replaced `.values[0]` scalar extraction with `.item()` to resolve NumPy `DeprecationWarning`.
- Replaced chained `fillna` assignment with direct assignment to resolve Pandas `DeprecationWarning`.
- Suppressed `FigureCanvasAgg UserWarning` from `plt.show()` in headless CI environments.
- Fixed Flake8 unterminated string literal issues across source files and unit tests.

### Removed
- Deprecated `im_calculator.py` and `slf_generator.py` modules (replaced by `imcalculator.py` and `slfgenerator.py`).

---

## v1.0.0

### Added or Changed
- Stable source code for vulnerability-toolkit
- Added AGPL v3 license
- Added CONTRIBUTORS.txt

### Removed
