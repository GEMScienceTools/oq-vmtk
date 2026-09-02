# Changelog

## [Unreleased]

### Added
- `modeller.__init__`: new optional `pinching4_params` (dict overriding Pinching4 pinching/damage parameters — `rDispP/N`, `rForceP/N`, `uForceP/N`, `gK1-4`/`gKLim`, `gD1-4`/`gDLim`, `gF1-4`/`gFLim`, `gE`, `dmgType`) and `minmax_multiplier` (scales the MinMax collapse-detection bound) arguments. Both default to today's hardcoded values (identical behaviour for `degradation=True` and `degradation=False` when omitted).
- `plotter.plot_vulnerability_function`: new optional `xlims` parameter (`[min, max]`) to restrict the plotted intensity range. Data arrays are filtered before plot creation so the violin axis is never distorted.

### Changed
- **Calibration reverted to a function-based, dependency-free design**: `calibration.py`'s class-based `calibration` API (mass/stiffness matrix assembly, eigenvalue modal analysis, iterative period-matching, and OpenSees SPO verification) is replaced by a lightweight `calibrate_model(nst, sdof_capacity, is_sos=False, is_frame=False, storey_heights=None, verbose=False)` function plus a private `_dynamic_properties()` helper, mirroring the mode-shape logic used in the `ppGlobal_PrepareThresholds.py` pipeline scripts. The first-mode shape is now either an assumed power law (frame buildings, `is_frame=True`, ≤ 12 storeys) or the first eigenvector of a simple tri-diagonal stiffness matrix (softened at the ground floor for soft-storey buildings, which always use the eigenvector shape regardless of `is_frame`). There is no more period-matching step and no more `openquake.vmtk.modeller`/OpenSeesPy dependency; `storey_heights` is now an inert pass-through carried into `metadata['storey_heights']`. The returned `metadata` dict is now `{'gamma_real', 'is_sos', 'is_frame', 'storey_heights'}` (previously ~17-24 keys including `T_target`, `Gamma`, `M_eff`, `shear_ratios`, `M`, `K`, and optional SPO verification arrays). The 5-tuple return shape `(floor_masses, storey_drifts, storey_forces, phi, metadata)` is unchanged, so demos and other callers that only use `floor_masses`/`storey_drifts`/`storey_forces` are unaffected. Unit tests, demos, and docs updated accordingly.
- `postprocessor.calculate_average_annual_loss` and `postprocessor.calculate_average_annual_damage_probability` merged into a single `postprocessor.calculate_risk(input_array, hazard_array, return_period=1, max_return_period=5000)` method. `input_array` accepts either a fragility curve (yielding the AADP) or a vulnerability curve (yielding the AALR); the integration logic is unchanged. Demos and docs updated accordingly.
- `plotter.plot_modes`: mode titles are now placed via `fig.text()` centred over both the X- and Y-displacement panels instead of being attached only to the left panel. Font sizes for node annotations, axis labels, tick marks, and panel titles, and the vertical spacing parameters (`hspace`, `top`), now scale adaptively with the number of grid rows and nodes per panel, preventing text and title overlap when plotting many modes.
- `plotter.plot_vulnerability_function`: x-axis tick label font size reduced to prevent overlap when many intensity levels are present.
- Demo `NonlinearTimeHistoryAnalysis`: introduction and section descriptions updated to match the concise style used across other demos; verbose bullet-point input/output argument lists removed.
- Demo `PushoverAnalysis`: cyclic pushover protocol updated to a two-repetition-per-level scheme (ATC-24 / FEMA 461 style), producing 20 full hysteresis loops instead of 11.
- Demo `StoreyLossFunctionApplication`: loss vs. IM fitting replaced with a power-law regression (OLS in log space, monotone by construction) paired with a Beta distribution dispersion model; COV estimated per quantile bin from the cloud scatter and propagated through the collapse-conditioning step using the law of total variance; summary tables of IML / mean loss / COV printed at both the non-conditioned and conditioned stages.
- `README.md`: all module and demo descriptions updated; badge URLs corrected (`.svg` suffix removed from shields.io GitHub badges; Zenodo DOI badge migrated to shields.io static endpoint); contributor shield URLs fixed; installation, license, citation, and references sections added.
- Demo `README.md` files: titles and descriptions corrected across `MultipleStripeAnalysis`, `FragilityAnalysis`, `IntensityMeasureProcessing`; `IntensityMeasureSelection/README.md` created; `demos/README.md` index updated to list all 13 demos.
- `slfgenerator.generate()`: `out[group]` now returns the empirical 16th/50th (median)/84th percentile of the storey loss ratio vs. EDP (`'slf_16th'`/`'slf'`/`'slf_84th'`) in place of a fitted regression curve; `'error_max'`/`'error_cum'` are removed. `cache[group]`'s empirical percentiles are now computed once and shared with `out` (previously duplicated); `'losses'`/`'slfs'`/`'fit_pars'`/`'accuracy'`/`'regression'` removed from `cache`. This is also a mean → median semantic shift for the primary curve, not just fitted → empirical.
- `slfgenerator`: internal component identity is now resolved by `Component ID` rather than row position — fragilities, damage states, repair costs, and correlation-tree lookups are all keyed by the actual ID. Component inventories and correlation trees no longer need sequential or matching-row-order IDs; a correlation tree may list components in any order. Previously, `validate_ds_dependence` silently mismatched components whenever `Component ID` deviated from `1..n` in row order — this is now fixed, not just guarded against.
- `slfgenerator.calculate_costs`: Normal-distributed repair-cost sampling switched from a rejection-sampling `while` loop to `scipy.stats.truncnorm`, vectorised per (component, realisation, damage-state) batch instead of one draw at a time. Same distribution, but eliminates a potential infinite loop (when `cov == 0` and `mu <= 0`, every rejected draw was identical) and runs roughly 100x faster at realistic realization counts.
- `plotter.plot_slf_model`: switched to loss-ratio units throughout (scatter, percentile band, and median line all sourced from `total_loss_storey_ratio`/`slf_16th`/`slf`/`slf_84th`); the "SLF - Fitted" line is removed (there is no longer a fitted curve). Multiple `out`/`cache` keys are now correctly overlaid on one set of axes instead of only the last key being rendered — previously a new figure/axes was created per key but only the last one was ever styled, shown, or saved. Colour cycling reordered so a 2-3 way comparison uses visually distinct colours instead of adjacent near-duplicate shades in the `gem` palette.
- Demo `StoreyLossFunctionGeneration`: restructured into Example A (independent components) and Example B (correlated components, via correlation trees on both the drift- and acceleration-sensitive inventories); the manual "recompute empirical stats" cells and `*_empirical.pkl` exports are removed now that `generate()` returns percentiles natively. The two correlation trees are now plain CSVs (`in/correlation_tree_psd.csv`, `in/correlation_tree_pfa.csv`, alongside the inventory CSVs they apply to) loaded with `pd.read_csv`, instead of being built in-notebook by a Python helper function.
- Demo `StoreyLossFunctionApplication`: re-themed from "Fitted vs. Empirical" to "Example A vs. Example B" throughout (demand interpolation, vulnerability fitting, collapse conditioning, AALR comparison).

### Removed
- `postprocessor.calculate_rotated_fragility` and the `fragility_rotation`/`rotation_percentile` options on `process_mca_results`, `process_msa_results`, and `process_ida_results` are removed (along with the `is_rotated`/`rotation_active`/`rotation_percentile` keys previously returned in their result dicts). Demo `FragilityAnalysis`'s "Method 4: rotation" is removed and the remaining methods renumbered 1–8; its "recommended method" plot now uses the lognormal + building-to-building + damage-state uncertainty variant.
- `postprocessor.calculate_vulnerability_function`: the `method='silva'` empirical COV option (Silva, 2019) is removed, along with the `method` parameter. Explicit law-of-total-variance uncertainty propagation is now always used when `uncertainty=True`.
- `slfgenerator`: all internal regression/curve-fitting machinery removed — `perform_regression`, `_fit_regression`, and `estimate_accuracy` methods, the `regression` constructor parameter, and the `fitting_model`/`fitting_parameters_model`/`fitted_loss_model` Pydantic scaffolding. `generate()` no longer fits a parametric curve to the simulated loss cloud.

### Fixed
- `modeller.do_nrha_analysis`/`do_nrha_analysis_sequences`: collapse detection relied solely on `ops.eleResponse(ele, 'deformation')`, which can freeze at its last valid value (below the MinMax limit) once a spring's material fully dies, while the true nodal displacement keeps diverging — a run could report "ANALYSIS SUCCESSFUL" with a physically absurd drift. Both methods now also cross-check the independently-computed nodal interstorey displacement against the MinMax limit every step (the same fix already present in `do_cpo_analysis`, now back-ported), and their `eleResponse` check inspects both X and Y deformation components under bidirectional loading instead of only X.
- `modeller.do_cpo_analysis`/`do_nrha_analysis`/`do_nrha_analysis_sequences`: the Python-side `minmax_limits` used for collapse detection hardcoded a `1.0×` multiplier instead of using `self.minmax_multiplier`, so the multiplier had no effect on when these methods reported collapse (it was already correctly applied to the actual OpenSees `MinMax` material bound). Now consistent throughout.
- `modeller.do_spo_analysis`: had no collapse detection at all (unlike `do_cpo_analysis`/`do_nrha_analysis`) — a spring failure could silently decay the base shear to ~0 while reporting success, or raise an unhandled `TypeError` if a spring were killed by OpenSees. Now performs the same nodal cross-check and `eleResponse`-based MinMax check as `do_cpo_analysis`, and returns a new `'conv_index'` key (0 = success, -1 = failure) in `spo_dict`.
- `modeller.do_cpo_analysis`/`do_spo_analysis`: the `eleResponse`-based MinMax check and (for SPO) the recorded `spo_disps_spring`/`spo_forces_spring` hardcoded deformation/force index `[0]`, ignoring `push_dir` — for `push_dir=2`/`3` these silently inspected/recorded the wrong DOF. Now use `push_dir - 1`.
- Demo `NonlinearTimeHistoryAnalysis`: its ground-motion record was already driving the calibrated 2-storey model's peak drift (0.0128) past the true ultimate storey capacity (0.0121) — previously silently reported as "ANALYSIS SUCCESSFUL" due to the `do_nrha_analysis` bug above; now correctly reports collapse. Re-executed to update the stored results/animation.
- Demo `PushoverAnalysis`: `disp_scale_factor` for the SPO example reduced from 15 to 14 — at 15, the pushover hit a genuine `DisplacementControl` snap-through near the softening branch (load factor spiking then collapsing to zero) that was previously recorded as valid data with no detection; `do_spo_analysis` now correctly catches this, so the target is reduced to stay within the well-behaved range while still tracing most of the softening branch.
- Demo `MultipleStripeAnalysis`: `damage_thresholds` (DS1-DS4) were arbitrary values unrelated to the calibrated model's actual capacity, with DS3 and DS4 both set *above* the governing (smaller-capacity) storey's ultimate drift ratio (~0.0065 rad). This was previously masked by the `do_nrha_analysis` collapse-detection bug above, which let a collapsing run's recorded drift balloon to physically absurd values (5-12%) that happened to still exceed the mis-set DS4 threshold; with collapse now correctly capped at the true capacity, genuinely-collapsed runs could fall short of DS3/DS4 entirely. `damage_thresholds` is now proportionally rescaled to `[0.00072, 0.00117, 0.00456, 0.00647]` so DS4 lands exactly at the governing ultimate capacity. Re-executed to update the fitted fragility functions and stored results.
- `docsrc/contents/pos/calculate_vulnerability_function.rst`: the Beta-distribution `α`/`β` moment-matching formula divided by `CoV²` directly; the actual variance used by both `plotter.plot_vulnerability_function` and `postprocessor.calculate_vulnerability_function` is `(CoV·μ)²` (missing a `μ²` factor). Corrected — the underlying code was already correct, only the documented formula was wrong.
- `plotter.plot_fragility_from_mca`: `title` parameter was silently ignored (variable was set but `ax.set_title()` was never called); title is now correctly applied to the figure.
- `plotter.plot_vulnerability_function`: Beta distribution sampling returned all-NaN when CoV was zero (division by zero produced `inf` parameters); now returns a point mass at the mean for the degenerate case.
- `slfgenerator.generate()`: `cache[group]['damage_states']` now correctly holds the group-sliced damage states instead of the full, ungrouped dictionary.
- `docsrc/contents/plo/plot_slf_model.rst` and `docsrc/contents/examples/storey_loss_function_generation.rst`: example code called `plotter.plot_slf_model` with a wrong keyword (`slf=` instead of `out=`) and/or without the required `xlims`/`ylims` arguments; either would have raised at runtime. Fixed.

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
