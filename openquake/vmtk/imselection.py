"""
Intensity Measure (IM) Selection Framework

Implements the Relative Sufficiency Measure (RSM) and associated efficiency /
proficiency metrics from:

    Ebrahimian, H. & Jalayer, F. (2021). "Selection of seismic intensity
    measures for prescribed limit states using alternative nonlinear dynamic
    analysis methods." Earthquake Engineering & Structural Dynamics, 50(5),
    1235–1258. https://doi.org/10.1002/eqe.3393

The RSM quantifies, in bits of information, how much more structural-demand
information IM₂ provides relative to IM₁.  A positive RSM(IM₂ vs IM₁) means
IM₂ is the more sufficient intensity measure.

Two nonlinear dynamic analysis procedures (NDAPs) are supported:
    - MCA  — Modified Cloud Analysis (``postprocessor.process_mca_results``)
    - IDA  — Incremental Dynamic Analysis (``postprocessor.process_ida_results``)
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Module-level numerical floor constants
# ---------------------------------------------------------------------------
_PDF_FLOOR = 1e-300   # floor for any probability density before log
_BETA_FLOOR = 1e-6    # floor for any dispersion value
_IM_FLOOR = 1e-10     # floor for IM values before log-transform


class imselection:
    """
    Intensity Measure selection tools based on the RSM framework.

    All methods are stateless; instantiate once and reuse freely::

        ims = imselection()
        eff = ims.compute_efficiency_mca(cloud_dict_pga)
        rsm = ims.compute_rsm_mca(cloud_dict_pga, cloud_dict_sa)
    """

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _lognormal_pdf(self, x, mu_ln, beta):
        """
        Lognormal probability density function.

        Parameters
        ----------
        x : array_like
            Values at which to evaluate the PDF (must be positive).
        mu_ln : array_like
            Mean of ln(x).
        beta : array_like
            Standard deviation of ln(x).  Clipped to ``_BETA_FLOOR``.

        Returns
        -------
        ndarray
            PDF values, floored at ``_PDF_FLOOR``.
        """
        x = np.maximum(np.asarray(x, dtype=float), _IM_FLOOR)
        beta = np.maximum(np.asarray(beta, dtype=float), _BETA_FLOOR)
        z = (np.log(x) - mu_ln) / beta
        pdf = norm.pdf(z) / (x * beta)
        return np.maximum(pdf, _PDF_FLOOR)

    def _pnoc_logistic(self, ln_im, alpha0, alpha1):
        """
        Probability of non-collapse from a logistic regression model.

        ``P(NoC | IM) = 1 − sigmoid(α₀ + α₁ · ln(IM))``

        Parameters
        ----------
        ln_im : array_like
            Natural log of the intensity measure values.
        alpha0 : float
            Logistic regression intercept.
        alpha1 : float
            Logistic regression slope on ln(IM).

        Returns
        -------
        ndarray
            P(NoC | IM) in [0, 1].
        """
        logit = np.clip(alpha0 + alpha1 * np.asarray(ln_im, dtype=float),
                        -500.0, 500.0)
        return 1.0 - 1.0 / (1.0 + np.exp(-logit))

    def _extract_ida_stats_at_demands(self, ida_dict, demand_values):
        """
        Interpolate IDA ensemble statistics and per-record IM at given demands.

        For each record k the method returns:
        * ``eta[k]``           — ensemble median IM at ``demand_values[k]``
        * ``beta[k]``          — ensemble dispersion (0.5·ln(p84/p16)) at that demand
        * ``im_per_record[k]`` — IM from record k's own IDA curve at ``demand_values[k]``

        A record is flagged as *valid* only when the demand falls within the
        tabulated EDP range, p84 > p16 > 0, and the per-record IM is positive.

        Parameters
        ----------
        ida_dict : dict
            Output of ``postprocessor.process_ida_results``.
        demand_values : array_like
            EDP level for each record, shape ``(n_records,)``.

        Returns
        -------
        eta : ndarray, shape (n_records,)
        beta : ndarray, shape (n_records,)
        im_per_record : ndarray, shape (n_records,)
        valid : ndarray of bool, shape (n_records,)
        """
        edp_axis = np.asarray(ida_dict['stats']['fitted_edps'])
        median_im_arr = np.asarray(ida_dict['stats']['median_im'])
        p16_im_arr = np.asarray(ida_dict['stats']['p16_im'])
        p84_im_arr = np.asarray(ida_dict['stats']['p84_im'])
        raw_curves = ida_dict['ida_inputs']['raw_curves']

        demand_values = np.asarray(demand_values, dtype=float)
        n = len(demand_values)

        eta = np.full(n, np.nan)
        beta = np.full(n, np.nan)
        im_per_record = np.full(n, np.nan)
        valid = np.zeros(n, dtype=bool)

        d_min, d_max = edp_axis[0], edp_axis[-1]

        for k, dk in enumerate(demand_values):
            if not (d_min <= dk <= d_max):
                continue
            eta[k] = np.interp(dk, edp_axis, median_im_arr)
            p16k = np.interp(dk, edp_axis, p16_im_arr)
            p84k = np.interp(dk, edp_axis, p84_im_arr)
            if p84k <= p16k or p16k <= 0:
                continue
            beta[k] = max(0.5 * np.log(p84k / p16k), _BETA_FLOOR)

            # Per-record IM at this demand
            curve_edp = np.asarray(raw_curves[k]['edp'])
            curve_im = np.asarray(raw_curves[k]['im'])
            if dk <= curve_edp[-1] and len(curve_edp) > 1:
                im_k = np.interp(dk, curve_edp, curve_im)
                if im_k > 0 and eta[k] > 0:
                    im_per_record[k] = im_k
                    valid[k] = True

        return eta, beta, im_per_record, valid

    def _validate_cloud_dict(self, cloud_dict, label='cloud_dict'):
        """
        Raise ``ValueError`` if *cloud_dict* is missing required regression keys.

        Parameters
        ----------
        cloud_dict : dict
            Output of ``postprocessor.process_mca_results``.
        label : str
            Name used in error messages.
        """
        if 'regression' not in cloud_dict:
            raise ValueError(
                f"{label} is missing the 'regression' key. "
                "Run process_mca_results first."
            )
        reg = cloud_dict['regression']
        required = ('b1', 'b0', 'sigma', 'alpha0', 'alpha1')
        missing = [k for k in required if reg.get(k) is None]
        if missing:
            raise ValueError(
                f"{label}['regression'] is missing keys: {missing}. "
                "Ensure postprocessor.py includes alpha0/alpha1 in regression."
            )

    # -----------------------------------------------------------------------
    # Efficiency
    # -----------------------------------------------------------------------

    def compute_efficiency_mca(self, cloud_dict):
        """
        Classic efficiency βD|IM from a Modified Cloud Analysis result.

        Efficiency is defined as the standard deviation of the residuals of the
        log-linear cloud regression, σ (sigma).  A lower value indicates a
        more efficient intensity measure.

        Parameters
        ----------
        cloud_dict : dict
            Output of ``postprocessor.process_mca_results``.

        Returns
        -------
        dict with keys:

        * ``'beta_D_given_IM'`` — regression residual sigma
        * ``'method'``          — ``'MCA'``
        """
        self._validate_cloud_dict(cloud_dict, 'cloud_dict')
        sigma = cloud_dict['regression']['sigma']
        return {'beta_D_given_IM': float(sigma), 'method': 'MCA'}

    def compute_efficiency_ida(self, ida_dict, ds_index=0):
        """
        Classic efficiency βD|IM from an Incremental Dynamic Analysis result.

        Uses the record-to-record dispersion of the fragility curve at the
        specified damage state.

        Parameters
        ----------
        ida_dict : dict
            Output of ``postprocessor.process_ida_results``.
        ds_index : int, optional
            Damage-state index (0 = first / least severe).  Default ``0``.

        Returns
        -------
        dict with keys:

        * ``'beta_D_given_IM'`` — sigma_record2record at damage state *ds_index*
        * ``'method'``          — ``'IDA'``
        """
        sigmas = ida_dict['fragility']['sigma_record2record']
        if ds_index >= len(sigmas):
            raise ValueError(
                f"ds_index={ds_index} out of range "
                f"(fragility has {len(sigmas)} damage states)."
            )
        return {'beta_D_given_IM': float(sigmas[ds_index]), 'method': 'IDA'}

    # -----------------------------------------------------------------------
    # Practicality
    # -----------------------------------------------------------------------

    def compute_practicality_mca(self, cloud_dict):
        """
        Practicality measure from a Modified Cloud Analysis result.

        Practicality = slope *b* from the log-linear regression
        ``log(EDP) = b₀ + b · log(IM)``.  A higher slope indicates a stronger
        IM–EDP correlation and therefore a more practical intensity measure.

        Parameters
        ----------
        cloud_dict : dict
            Output of ``postprocessor.process_mca_results``.

        Returns
        -------
        dict with keys:

        * ``'b_slope'`` — regression slope *b* (higher is better)
        * ``'method'``  — ``'MCA'``
        """
        self._validate_cloud_dict(cloud_dict, 'cloud_dict')
        return {'b_slope': float(cloud_dict['regression']['b1']), 'method': 'MCA'}

    def compute_practicality_ida(self, ida_dict):
        """
        Practicality measure from an Incremental Dynamic Analysis result.

        Estimated as the OLS slope of ``log(EDP)`` on ``log(IM)`` along the
        median IDA curve, mirroring the MCA log-linear regression convention.
        A higher slope indicates a stronger IM–EDP relationship.

        Parameters
        ----------
        ida_dict : dict
            Output of ``postprocessor.process_ida_results``.

        Returns
        -------
        dict with keys:

        * ``'b_slope'`` — log-log slope (higher is better); NaN if the median
          curve has fewer than 2 positive-valued points
        * ``'method'``  — ``'IDA'``
        """
        edp_axis  = np.asarray(ida_dict['stats']['fitted_edps'])
        median_im = np.asarray(ida_dict['stats']['median_im'])
        valid = (edp_axis > _IM_FLOOR) & (median_im > _IM_FLOOR)
        if valid.sum() < 2:
            return {'b_slope': float(np.nan), 'method': 'IDA'}
        slope, _ = np.polyfit(np.log(median_im[valid]), np.log(edp_axis[valid]), 1)
        return {'b_slope': float(slope), 'method': 'IDA'}

    # -----------------------------------------------------------------------
    # Proficiency
    # -----------------------------------------------------------------------

    def compute_proficiency_mca(self, cloud_dict, ds_index=0):
        """
        Proficiency measure βIM|DCR=1 from a Modified Cloud Analysis result.

        Proficiency is estimated as ``0.5 · ln(im84 / im16)`` where im16 and
        im84 are the IM values at which the fragility curve reaches 16 % and
        84 % exceedance probability respectively.

        An additional analytical estimate ``'beta_formula'`` = σ / b is
        returned as a cross-check (it equals the proficiency exactly when no
        records collapse, i.e., under the plain Cloud Analysis assumption).

        Parameters
        ----------
        cloud_dict : dict
            Output of ``postprocessor.process_mca_results``.
        ds_index : int, optional
            Damage-state index.  Default ``0``.

        Returns
        -------
        dict with keys:

        * ``'beta_IM_given_DCRLS1'`` — proficiency (NaN if curve does not
          reach 0.16 or 0.84)
        * ``'beta_formula'``         — analytical cross-check σ / b
        * ``'im16'``                 — IM at 16 % exceedance (NaN if not reached)
        * ``'im84'``                 — IM at 84 % exceedance (NaN if not reached)
        * ``'method'``               — ``'MCA'``
        """
        self._validate_cloud_dict(cloud_dict, 'cloud_dict')

        intensities = np.asarray(cloud_dict['fragility']['intensities'])
        poes = np.asarray(cloud_dict['fragility']['poes'])
        if poes.ndim == 1:
            poe_curve = poes
        else:
            if ds_index >= poes.shape[1]:
                raise ValueError(
                    f"ds_index={ds_index} out of range "
                    f"(fragility has {poes.shape[1]} damage states)."
                )
            poe_curve = poes[:, ds_index]

        im16 = float(np.nan)
        im84 = float(np.nan)
        beta_prof = float(np.nan)

        if poe_curve.min() < 0.16 < poe_curve.max():
            im16 = float(np.interp(0.16, poe_curve, intensities))
        else:
            warnings.warn(
                "Fragility curve does not reach 16 % exceedance for "
                f"ds_index={ds_index}. Returning NaN for im16."
            )

        if poe_curve.min() < 0.84 < poe_curve.max():
            im84 = float(np.interp(0.84, poe_curve, intensities))
        else:
            warnings.warn(
                "Fragility curve does not reach 84 % exceedance for "
                f"ds_index={ds_index}. Returning NaN for im84."
            )

        if np.isfinite(im16) and np.isfinite(im84) and im84 > im16 > 0:
            beta_prof = float(0.5 * np.log(im84 / im16))

        sigma = cloud_dict['regression']['sigma']
        b = cloud_dict['regression']['b1']
        beta_formula = float(sigma / b) if b != 0 else float(np.nan)

        return {
            'beta_IM_given_DCRLS1': beta_prof,
            'beta_formula': beta_formula,
            'im16': im16,
            'im84': im84,
            'method': 'MCA',
        }

    def compute_proficiency_ida(self, ida_dict, ds_index=0):
        """
        Proficiency measure βIM|DCR=1 from an Incremental Dynamic Analysis result.

        Proficiency is estimated as ``0.5 · ln(im84 / im16)`` where im16 and
        im84 are the 16th and 84th percentile IDA curves evaluated at the
        damage threshold for the specified damage state.

        Parameters
        ----------
        ida_dict : dict
            Output of ``postprocessor.process_ida_results``.
        ds_index : int, optional
            Damage-state index.  Default ``0``.

        Returns
        -------
        dict with keys:

        * ``'beta_IM_given_DCRLS1'`` — proficiency (NaN if p16 ≥ p84)
        * ``'im16'``                 — IM at 16th percentile for this DS
        * ``'im84'``                 — IM at 84th percentile for this DS
        * ``'method'``               — ``'IDA'``
        """
        damage_thresholds = ida_dict['ida_inputs']['damage_thresholds']
        if ds_index >= len(damage_thresholds):
            raise ValueError(
                f"ds_index={ds_index} out of range "
                f"(IDA has {len(damage_thresholds)} damage states)."
            )

        edp_axis = np.asarray(ida_dict['stats']['fitted_edps'])
        p16_arr = np.asarray(ida_dict['stats']['p16_im'])
        p84_arr = np.asarray(ida_dict['stats']['p84_im'])
        ds_edp = damage_thresholds[ds_index]

        im16 = float(np.interp(ds_edp, edp_axis, p16_arr))
        im84 = float(np.interp(ds_edp, edp_axis, p84_arr))

        if im84 <= im16 or im16 <= 0:
            warnings.warn(
                f"p84 ≤ p16 at damage threshold for ds_index={ds_index}. "
                "Returning NaN for proficiency."
            )
            return {
                'beta_IM_given_DCRLS1': float(np.nan),
                'im16': im16,
                'im84': im84,
                'method': 'IDA',
            }

        beta_prof = float(0.5 * np.log(im84 / im16))
        return {
            'beta_IM_given_DCRLS1': beta_prof,
            'im16': im16,
            'im84': im84,
            'method': 'IDA',
        }

    # -----------------------------------------------------------------------
    # RSM — Modified Cloud Analysis (Eq. 5)
    # -----------------------------------------------------------------------

    def compute_rsm_mca(self, cloud_dict_im1, cloud_dict_im2):
        """
        Relative Sufficiency Measure between two IMs using Modified Cloud Analysis.

        Implements Equation 5 of Ebrahimian & Jalayer (2021).

        A positive RSM value means IM₂ is more sufficient than IM₁ for
        characterising structural demand.

        **API contract:** *cloud_dict_im1* and *cloud_dict_im2* must have been
        computed from **the same N ground-motion records, in the same order**,
        with the **same** ``censored_limit`` and ``lower_limit`` parameters.
        Under these conditions the EDP-based collapse flag and lower-EDP filter
        are identical for both IMs, so ``raw_data['im_nc']`` arrays are
        positionally aligned.

        Parameters
        ----------
        cloud_dict_im1 : dict
            MCA result for IM₁ (reference IM).
        cloud_dict_im2 : dict
            MCA result for IM₂ (candidate IM).

        Returns
        -------
        dict with keys:

        * ``'rsm'``            — scalar RSM in bits (positive → IM₂ better)
        * ``'rsm_per_record'`` — ndarray of per-record log₂ ratios, shape (N,)
        * ``'n_records'``      — number of non-collapse records used
        * ``'method'``         — ``'MCA'``
        * ``'params_im1'``     — regression parameters used for IM₁
        * ``'params_im2'``     — regression parameters used for IM₂
        """
        self._validate_cloud_dict(cloud_dict_im1, 'cloud_dict_im1')
        self._validate_cloud_dict(cloud_dict_im2, 'cloud_dict_im2')

        # Extract regression parameters
        def _params(cd):
            r = cd['regression']
            return {
                'b0': float(r['b0']),
                'b1': float(r['b1']),
                'sigma': max(float(r['sigma']), _BETA_FLOOR),
                'alpha0': float(r['alpha0']),
                'alpha1': float(r['alpha1']),
            }

        p1 = _params(cloud_dict_im1)
        p2 = _params(cloud_dict_im2)

        # Non-collapse records (lower_limit already applied in postprocessor)
        edp_nc = np.asarray(cloud_dict_im1['raw_data']['edp_nc'])
        im1_nc = np.asarray(cloud_dict_im1['raw_data']['im_nc'])
        im2_nc = np.asarray(cloud_dict_im2['raw_data']['im_nc'])

        n = len(edp_nc)
        if n == 0:
            raise ValueError(
                "No non-collapse records found in cloud_dict_im1['raw_data']. "
                "Cannot compute RSM."
            )
        if len(im2_nc) != n:
            raise ValueError(
                f"cloud_dict_im1 has {n} non-collapse records but "
                f"cloud_dict_im2 has {len(im2_nc)}. "
                "Both dicts must be derived from the same record set."
            )

        # Clip IM values
        im1_nc = np.maximum(im1_nc, _IM_FLOOR)
        im2_nc = np.maximum(im2_nc, _IM_FLOOR)
        ln_im1 = np.log(im1_nc)
        ln_im2 = np.log(im2_nc)
        ln_edp = np.log(np.maximum(edp_nc, _IM_FLOOR))

        # Standardised residuals
        z1 = (ln_edp - p1['b0'] - p1['b1'] * ln_im1) / p1['sigma']
        z2 = (ln_edp - p2['b0'] - p2['b1'] * ln_im2) / p2['sigma']

        # P(NoC | IM)
        pnoc1 = self._pnoc_logistic(ln_im1, p1['alpha0'], p1['alpha1'])
        pnoc2 = self._pnoc_logistic(ln_im2, p2['alpha0'], p2['alpha1'])

        # Per-record likelihood ratio (the 1/(edp·β) factors appear in both
        # numerator and denominator so only β1/β2 survives)
        phi1 = np.maximum(norm.pdf(z1), _PDF_FLOOR)
        phi2 = np.maximum(norm.pdf(z2), _PDF_FLOOR)

        ratio = (phi2 * pnoc2 * p1['sigma']) / (phi1 * pnoc1 * p2['sigma'])
        ratio = np.maximum(ratio, _PDF_FLOOR)

        log2_ratio = np.log2(ratio)

        # Screen bad values
        bad = ~np.isfinite(log2_ratio)
        n_bad = int(bad.sum())
        if n_bad > 0:
            warnings.warn(
                f"{n_bad} record(s) produced non-finite log₂ ratio in "
                "compute_rsm_mca and were excluded from the mean."
            )
            log2_ratio = log2_ratio[~bad]

        rsm = float(np.mean(log2_ratio)) if len(log2_ratio) > 0 else float(np.nan)

        return {
            'rsm': rsm,
            'rsm_per_record': log2_ratio,
            'n_records': n - n_bad,
            'method': 'MCA',
            'params_im1': p1,
            'params_im2': p2,
        }

    # -----------------------------------------------------------------------
    # RSM — Incremental Dynamic Analysis (Eq. 8)
    # -----------------------------------------------------------------------

    def compute_rsm_ida(self, ida_dict_im1, ida_dict_im2):
        """
        Relative Sufficiency Measure between two IMs using Incremental Dynamic
        Analysis.

        Implements Equation 8 of Ebrahimian & Jalayer (2021).

        **API contract:** *ida_dict_im1* and *ida_dict_im2* must have been
        computed from **the same N ground-motion records in the same order**.

        Parameters
        ----------
        ida_dict_im1 : dict
            IDA result for IM₁ (reference IM).
        ida_dict_im2 : dict
            IDA result for IM₂ (candidate IM).

        Returns
        -------
        dict with keys:

        * ``'rsm'``            — scalar RSM in bits (positive → IM₂ better)
        * ``'rsm_per_record'`` — ndarray of per-record log₂ ratios (valid only)
        * ``'n_records'``      — total number of records
        * ``'n_valid'``        — records where IDA interpolation succeeded
        * ``'method'``         — ``'IDA'``
        """
        n1 = ida_dict_im1['ida_inputs']['n_records']
        n2 = ida_dict_im2['ida_inputs']['n_records']
        if n1 != n2:
            raise ValueError(
                f"ida_dict_im1 has {n1} records but ida_dict_im2 has {n2}. "
                "Both dicts must be derived from the same record set."
            )
        n_records = n1

        # Use IM1's max demand per record as the common demand level, clipped
        # to the fitted EDP range.  Raw IDA curves can reach collapse EDPs well
        # beyond the fitted axis (edp_range[-1]); without clipping every record
        # would fail the range check in _extract_ida_stats_at_demands.
        raw1 = ida_dict_im1['ida_inputs']['raw_curves']
        edp_axis = np.asarray(ida_dict_im1['stats']['fitted_edps'])
        demand_values = np.clip(
            np.array([raw1[k]['edp'][-1] for k in range(n_records)]),
            edp_axis[0],
            edp_axis[-1],
        )

        # Ensemble stats + per-record IM at the demand level
        eta1, beta1, im1_per_rec, valid1 = self._extract_ida_stats_at_demands(
            ida_dict_im1, demand_values
        )
        eta2, beta2, im2_per_rec, valid2 = self._extract_ida_stats_at_demands(
            ida_dict_im2, demand_values
        )

        # P(NoC | IM) from collapse fragility (last DS = collapse)
        try:
            theta_c1 = ida_dict_im1['fragility']['medians'][-1]
            beta_c1 = ida_dict_im1['fragility']['sigma_record2record'][-1]
            theta_c2 = ida_dict_im2['fragility']['medians'][-1]
            beta_c2 = ida_dict_im2['fragility']['sigma_record2record'][-1]
            use_fragility = (
                theta_c1 > 0 and beta_c1 > 0
                and theta_c2 > 0 and beta_c2 > 0
            )
        except (KeyError, IndexError, TypeError):
            use_fragility = False

        valid = valid1 & valid2
        n_valid = int(valid.sum())
        if n_valid == 0:
            warnings.warn(
                "No valid records for IDA RSM computation "
                "(IDA interpolation failed for all records)."
            )
            return {
                'rsm': float(np.nan),
                'rsm_per_record': np.array([]),
                'n_records': n_records,
                'n_valid': 0,
                'method': 'IDA',
            }

        log2_ratio = np.full(n_records, np.nan)

        for k in range(n_records):
            if not valid[k]:
                continue

            im1k = max(float(im1_per_rec[k]), _IM_FLOOR)
            im2k = max(float(im2_per_rec[k]), _IM_FLOOR)
            eta1k = max(float(eta1[k]), _IM_FLOOR)
            eta2k = max(float(eta2[k]), _IM_FLOOR)
            beta1k = max(float(beta1[k]), _BETA_FLOOR)
            beta2k = max(float(beta2[k]), _BETA_FLOOR)

            if use_fragility:
                pnoc1k = max(
                    1.0 - norm.cdf(np.log(im1k / theta_c1) / beta_c1), _PDF_FLOOR
                )
                pnoc2k = max(
                    1.0 - norm.cdf(np.log(im2k / theta_c2) / beta_c2), _PDF_FLOOR
                )
            else:
                pnoc1k = 1.0
                pnoc2k = 1.0

            z1k = (np.log(im1k) - np.log(eta1k)) / beta1k
            z2k = (np.log(im2k) - np.log(eta2k)) / beta2k

            phi1k = max(norm.pdf(z1k), _PDF_FLOOR)
            phi2k = max(norm.pdf(z2k), _PDF_FLOOR)

            # Eq. 8 ratio: includes 1/(im·β) normalisation terms
            num = phi2k * pnoc2k * im1k * beta1k
            den = phi1k * pnoc1k * im2k * beta2k
            ratio_k = max(num / den, _PDF_FLOOR)
            log2_ratio[k] = np.log2(ratio_k)

        valid_log2 = log2_ratio[valid]
        bad = ~np.isfinite(valid_log2)
        n_bad = int(bad.sum())
        if n_bad > 0:
            warnings.warn(
                f"{n_bad} valid record(s) produced non-finite log₂ ratio in "
                "compute_rsm_ida and were excluded from the mean."
            )
            valid_log2 = valid_log2[~bad]

        rsm = float(np.mean(valid_log2)) if len(valid_log2) > 0 else float(np.nan)

        return {
            'rsm': rsm,
            'rsm_per_record': valid_log2,
            'n_records': n_records,
            'n_valid': n_valid - n_bad,
            'method': 'IDA',
        }

    # -----------------------------------------------------------------------
    # RSM — General / user-supplied PDF callables (Eq. 1)
    # -----------------------------------------------------------------------

    def compute_rsm_general(self, demands, im1_values, im2_values,
                            pdf_func_im1, pdf_func_im2):
        """
        General Relative Sufficiency Measure using user-supplied PDF callables.

        Implements Equation 1 of Ebrahimian & Jalayer (2021).  Useful for
        custom demand models or non-lognormal formulations.

        Parameters
        ----------
        demands : array_like
            EDP value for each record, shape ``(N,)``.
        im1_values : array_like
            IM₁ value for each record, shape ``(N,)``.
        im2_values : array_like
            IM₂ value for each record, shape ``(N,)``.
        pdf_func_im1 : callable
            ``f(d_k, im1_k) → float`` — probability density of demand *d_k*
            given IM₁ = *im1_k*.
        pdf_func_im2 : callable
            ``f(d_k, im2_k) → float`` — same for IM₂.

        Returns
        -------
        dict with keys:

        * ``'rsm'``            — scalar RSM in bits
        * ``'rsm_per_record'`` — ndarray of per-record log₂ ratios
        * ``'n_records'``      — total records
        * ``'n_valid'``        — records with finite log₂ ratio
        * ``'method'``         — ``'general'``
        """
        demands = np.asarray(demands, dtype=float)
        im1_values = np.asarray(im1_values, dtype=float)
        im2_values = np.asarray(im2_values, dtype=float)
        n = len(demands)

        log2_ratio = np.full(n, np.nan)
        for k in range(n):
            f1 = max(float(pdf_func_im1(demands[k], im1_values[k])), _PDF_FLOOR)
            f2 = max(float(pdf_func_im2(demands[k], im2_values[k])), _PDF_FLOOR)
            log2_ratio[k] = np.log2(f2 / f1)

        bad = ~np.isfinite(log2_ratio)
        n_bad = int(bad.sum())
        if n_bad > 0:
            warnings.warn(
                f"{n_bad} record(s) produced non-finite log₂ ratio in "
                "compute_rsm_general and were excluded from the mean."
            )

        valid_log2 = log2_ratio[~bad]
        rsm = float(np.mean(valid_log2)) if len(valid_log2) > 0 else float(np.nan)

        return {
            'rsm': rsm,
            'rsm_per_record': valid_log2,
            'n_records': n,
            'n_valid': n - n_bad,
            'method': 'general',
        }

    # -----------------------------------------------------------------------
    # Orchestrator
    # -----------------------------------------------------------------------

    def compare_ims(self, im_results_dict, analysis_type='MCA', metric='all',
                    reference_im_key=None, damage_threshold_index=0):
        """
        Compare N ≥ 2 intensity measures by efficiency, proficiency,
        practicality, and RSM.

        Parameters
        ----------
        im_results_dict : dict
            Mapping ``{im_name: cloud_dict}`` (for MCA) or
            ``{im_name: ida_dict}`` (for IDA).
        analysis_type : {'MCA', 'IDA'}
            Which NDAP was used to generate the results.
        metric : {'all', 'efficiency', 'proficiency', 'practicality', 'rsm'}
            Which metrics to compute.  ``'all'`` computes all four.
        reference_im_key : str or None
            IM name to use as IM₁ in RSM pairwise comparisons.  If ``None``,
            the first key in *im_results_dict* is used.
        damage_threshold_index : int
            Damage-state index for proficiency computation.  Default ``0``.

        Returns
        -------
        dict with keys:

        * ``'ranking'``          — ``pd.DataFrame`` with columns im_name,
          efficiency, proficiency, practicality, rsm_vs_reference,
          rank_efficiency, rank_proficiency, rank_practicality, rank_rsm
        * ``'rsm_matrix'``       — ``dict[str, dict[str, float]]`` where
          ``rsm_matrix[im_i][im_j]`` = RSM(IM_j vs IM_i)
        * ``'metric'``           — the *metric* argument used
        * ``'analysis_type'``    — the *analysis_type* argument used
        * ``'reference_im_key'`` — the IM used as IM₁ for RSM

        Raises
        ------
        ValueError
            If fewer than 2 IMs are supplied or *analysis_type* is invalid.
        """
        if analysis_type not in ('MCA', 'IDA'):
            raise ValueError(
                f"analysis_type must be 'MCA' or 'IDA', got '{analysis_type}'."
            )
        im_names = list(im_results_dict.keys())
        if len(im_names) < 2:
            raise ValueError(
                "compare_ims requires at least 2 intensity measures; "
                f"got {len(im_names)}."
            )
        if reference_im_key is None:
            reference_im_key = im_names[0]
        if reference_im_key not in im_results_dict:
            raise ValueError(
                f"reference_im_key='{reference_im_key}' not found in "
                "im_results_dict."
            )

        do_eff  = metric in ('all', 'efficiency')
        do_prof = metric in ('all', 'proficiency')
        do_prac = metric in ('all', 'practicality')
        do_rsm  = metric in ('all', 'rsm')

        rows = []
        rsm_matrix = {n: {} for n in im_names}

        for im_name in im_names:
            res = im_results_dict[im_name]
            eff_val  = float(np.nan)
            prof_val = float(np.nan)
            prac_val = float(np.nan)

            if do_eff:
                if analysis_type == 'MCA':
                    eff_val = self.compute_efficiency_mca(res)['beta_D_given_IM']
                else:
                    eff_val = self.compute_efficiency_ida(
                        res, ds_index=damage_threshold_index
                    )['beta_D_given_IM']

            if do_prof:
                if analysis_type == 'MCA':
                    prof_val = self.compute_proficiency_mca(
                        res, ds_index=damage_threshold_index
                    )['beta_IM_given_DCRLS1']
                else:
                    prof_val = self.compute_proficiency_ida(
                        res, ds_index=damage_threshold_index
                    )['beta_IM_given_DCRLS1']

            if do_prac:
                if analysis_type == 'MCA':
                    prac_val = self.compute_practicality_mca(res)['b_slope']
                else:
                    prac_val = self.compute_practicality_ida(res)['b_slope']

            rows.append({
                'im_name':      im_name,
                'efficiency':   eff_val,
                'proficiency':  prof_val,
                'practicality': prac_val,
            })

        # RSM matrix
        if do_rsm:
            ref_res = im_results_dict[reference_im_key]
            for im_name in im_names:
                cand_res = im_results_dict[im_name]
                for ref_name in im_names:
                    ref_res_inner = im_results_dict[ref_name]
                    try:
                        if analysis_type == 'MCA':
                            r = self.compute_rsm_mca(ref_res_inner, cand_res)
                        else:
                            r = self.compute_rsm_ida(ref_res_inner, cand_res)
                        rsm_matrix[ref_name][im_name] = r['rsm']
                    except Exception as exc:  # noqa: BLE001
                        warnings.warn(
                            f"RSM({im_name} vs {ref_name}) failed: {exc}"
                        )
                        rsm_matrix[ref_name][im_name] = float(np.nan)

            # RSM vs reference IM
            ref_res = im_results_dict[reference_im_key]  # noqa: F841
            for row in rows:
                row['rsm_vs_reference'] = rsm_matrix[reference_im_key].get(
                    row['im_name'], float(np.nan)
                )
        else:
            for row in rows:
                row['rsm_vs_reference'] = float(np.nan)

        df = pd.DataFrame(rows)

        # Ranks (lower beta = better efficiency/proficiency; higher RSM = better)
        if do_eff:
            df['rank_efficiency'] = (
                df['efficiency'].rank(method='min', ascending=True).astype(int)
            )
        else:
            df['rank_efficiency'] = np.nan

        if do_prof:
            df['rank_proficiency'] = (
                df['proficiency'].rank(method='min', ascending=True).astype(int)
            )
        else:
            df['rank_proficiency'] = np.nan

        if do_prac:
            df['rank_practicality'] = (
                df['practicality'].rank(method='min', ascending=False).astype(int)
            )
        else:
            df['rank_practicality'] = np.nan

        if do_rsm:
            df['rank_rsm'] = (
                df['rsm_vs_reference']
                .rank(method='min', ascending=False).astype(int)
            )
        else:
            df['rank_rsm'] = np.nan

        return {
            'ranking': df,
            'rsm_matrix': rsm_matrix,
            'metric': metric,
            'analysis_type': analysis_type,
            'reference_im_key': reference_im_key,
        }
