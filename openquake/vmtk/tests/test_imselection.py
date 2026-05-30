"""
Unit tests for openquake.vmtk.im_selection
"""

import math
import unittest
import warnings

import numpy as np

from openquake.vmtk.imselection import imselection, _PDF_FLOOR


# ---------------------------------------------------------------------------
# Synthetic fixture factories
# ---------------------------------------------------------------------------

def _make_synthetic_cloud_dict(b1=1.2, b0=-3.5, sigma=0.4,
                               alpha0=-5.0, alpha1=3.0,
                               n=80, seed=0, add_collapse=False):
    """
    Build a minimal cloud_dict compatible with process_mca_results
    output without running the full postprocessor stack.

    Parameters
    ----------
    b1, b0, sigma : float
        Power-law regression parameters  ln(EDP) = b0 + b1·ln(IM) + ε.
    alpha0, alpha1 : float
        Logistic collapse model  P(C|IM) = sigmoid(α₀ + α₁·ln(IM)).
    n : int
        Number of non-collapse records.
    seed : int
        NumPy random seed for reproducibility.
    add_collapse : bool
        If True, add a few synthetic collapse IM values.
    """
    rng = np.random.default_rng(seed)
    # IM values spanning one order of magnitude
    im_nc = np.exp(rng.uniform(np.log(0.05), np.log(0.5), n))
    # EDP from power-law + lognormal scatter
    edp_nc = np.exp(b0 + b1 * np.log(im_nc) + rng.normal(0, sigma, n))

    im_c = np.exp(rng.uniform(np.log(0.3), np.log(1.0), 5)) if add_collapse else np.array([])

    # Fragility: one damage state — lognormal CDF fitted to make proficiency
    # tests deterministic.
    intensities = np.geomspace(0.01, 2.0, 50)
    from scipy.stats import norm
    # DS1: median = 0.15 g, sigma = sigma
    poes = norm.cdf(np.log(intensities / 0.15) / sigma).reshape(-1, 1)

    return {
        'regression': {
            'b1': b1,
            'b0': b0,
            'sigma': sigma,
            'alpha0': alpha0,
            'alpha1': alpha1,
            'fitted_x': np.log(intensities),
            'fitted_y': b0 + b1 * np.log(intensities),
        },
        'fragility': {
            'intensities': intensities,
            'poes': poes,
            'medians': [0.15],
            'sigma_record2record': [sigma],
        },
        'bootstraps': {
            'b1': np.full(200, b1),
            'a': np.exp(np.full(200, b0)),
            'sigma_rr': np.full(200, sigma),
            'alpha0': np.full(200, alpha0),
            'alpha1': np.full(200, alpha1),
        },
        'raw_data': {
            'im_nc': im_nc,
            'edp_nc': edp_nc,
            'im_c': im_c,
        },
    }


def _make_synthetic_ida_dict(n_records=30, n_steps=20, seed=1,
                             b1=1.2, sigma=0.4):
    """
    Build a minimal ida_dict compatible with postprocessor.process_ida_results
    output.

    IDA curves are generated as power-law: EDP = a·IM^b1 with lognormal
    scatter.  The ensemble stats (median, p16, p84) are derived analytically.
    """
    rng = np.random.default_rng(seed)
    im_levels = np.geomspace(0.02, 1.0, n_steps)
    edp_range = np.geomspace(1e-4, 0.15, 100)
    a = np.exp(-3.5)  # corresponds to b0 = -3.5

    raw_curves = []
    for _ in range(n_records):
        # Per-record IDA curve with record-to-record variability
        ln_a_rec = np.log(a) + rng.normal(0, sigma)
        edps = np.exp(ln_a_rec + b1 * np.log(im_levels))
        # Ensure monotonic (structural resurrection)
        edps = np.maximum.accumulate(edps)
        raw_curves.append({'im': im_levels.copy(), 'edp': edps})

    # Ensemble IDA stats: IM at each EDP level
    # For lognormal IDA: median IM at EDP d = (d/a)^(1/b1)
    median_im = (edp_range / a) ** (1.0 / b1)
    p16_im = median_im * np.exp(-sigma / b1)
    p84_im = median_im * np.exp(sigma / b1)

    damage_thresholds = [0.005, 0.02]
    intensities = np.geomspace(0.01, 2.0, 50)
    from scipy.stats import norm
    poes = np.column_stack([
        norm.cdf(np.log(intensities / (dt / a) ** (1 / b1)) / (sigma / b1))
        for dt in damage_thresholds
    ])
    thetas = [(dt / a) ** (1 / b1) for dt in damage_thresholds]
    sigmas_r2r = [sigma / b1] * len(damage_thresholds)

    return {
        'ida_inputs': {
            'target_edps': edp_range,
            'raw_curves': raw_curves,
            'damage_thresholds': damage_thresholds,
            'im_matrix': np.tile(im_levels, (n_records, 1)),
            'n_records': n_records,
            'im_max_analyzed': im_levels[-1],
        },
        'fragility': {
            'fragility_method': 'mle_ida_censored',
            'intensities': intensities,
            'poes': poes,
            'medians': thetas,
            'sigma_record2record': sigmas_r2r,
            'sigma_build2build': [0.3] * len(damage_thresholds),
            'sigma_ds': [0.2] * len(damage_thresholds),
            'betas_total': [0.5] * len(damage_thresholds),
            'rotation_active': False,
            'rotation_percentile': None,
        },
        'stats': {
            'fitted_edps': edp_range,
            'median_im': median_im,
            'p16_im': p16_im,
            'p84_im': p84_im,
        },
    }


# ---------------------------------------------------------------------------
# TestHelperMethods
# ---------------------------------------------------------------------------

class TestHelperMethods(unittest.TestCase):

    def setUp(self):
        self.ims = imselection()

    def test_lognormal_pdf_at_median(self):
        """PDF at the median (x = exp(mu_ln)) should equal phi(0)/(x*beta)."""
        mu_ln = 0.5
        beta = 0.3
        x = math.exp(mu_ln)
        from scipy.stats import norm
        expected = norm.pdf(0.0) / (x * beta)
        result = self.ims._lognormal_pdf(x, mu_ln, beta)
        self.assertAlmostEqual(float(result), expected, places=10)

    def test_lognormal_pdf_floor(self):
        """PDF far from the median is floored at _PDF_FLOOR."""
        # x very far from mu_ln → PDF extremely small → should be floored
        pdf = self.ims._lognormal_pdf(1e-6, 10.0, 0.1)
        self.assertGreaterEqual(float(pdf), _PDF_FLOOR)

    def test_pnoc_logistic_overflow_guard(self):
        """Logistic model must not raise overflow for extreme ln(IM) values."""
        ln_im = np.array([-1000.0, 0.0, 1000.0])
        result = self.ims._pnoc_logistic(ln_im, -5.0, 3.0)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_pnoc_zero_collapse_limit(self):
        """When alpha1 = 0 and alpha0 → -∞, P(NoC|IM) → 1 for all IM."""
        result = self.ims._pnoc_logistic(np.array([0.0, 1.0]), -500.0, 0.0)
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_validate_cloud_dict_missing_key(self):
        """Missing 'regression' key raises ValueError."""
        with self.assertRaises(ValueError):
            self.ims._validate_cloud_dict({}, label='test')

    def test_validate_cloud_dict_none_values(self):
        """None values in regression dict raise ValueError."""
        cd = {'regression': {'b1': None, 'b0': -3.5, 'sigma': 0.4,
                             'alpha0': -5.0, 'alpha1': 3.0}}
        with self.assertRaises(ValueError):
            self.ims._validate_cloud_dict(cd)


# ---------------------------------------------------------------------------
# TestEfficiencyMCA
# ---------------------------------------------------------------------------

class TestEfficiencyMCA(unittest.TestCase):

    def setUp(self):
        self.ims = imselection()
        self.cd = _make_synthetic_cloud_dict(sigma=0.35)

    def test_returns_correct_sigma(self):
        result = self.ims.compute_efficiency_mca(self.cd)
        self.assertAlmostEqual(result['beta_D_given_IM'], 0.35, places=10)
        self.assertEqual(result['method'], 'MCA')

    def test_raises_on_none_regression(self):
        cd = {'regression': {'b1': None, 'b0': -3.5, 'sigma': None,
                             'alpha0': None, 'alpha1': None}}
        with self.assertRaises(ValueError):
            self.ims.compute_efficiency_mca(cd)

    def test_raises_when_regression_missing(self):
        with self.assertRaises(ValueError):
            self.ims.compute_efficiency_mca({'fragility': {}})


# ---------------------------------------------------------------------------
# TestEfficiencyIDA
# ---------------------------------------------------------------------------

class TestEfficiencyIDA(unittest.TestCase):

    def setUp(self):
        self.ims = imselection()
        self.ida = _make_synthetic_ida_dict(sigma=0.4)

    def test_correct_sigma_ds0(self):
        result = self.ims.compute_efficiency_ida(self.ida, ds_index=0)
        expected = self.ida['fragility']['sigma_record2record'][0]
        self.assertAlmostEqual(result['beta_D_given_IM'], expected, places=10)
        self.assertEqual(result['method'], 'IDA')

    def test_correct_sigma_ds1(self):
        result = self.ims.compute_efficiency_ida(self.ida, ds_index=1)
        expected = self.ida['fragility']['sigma_record2record'][1]
        self.assertAlmostEqual(result['beta_D_given_IM'], expected, places=10)

    def test_out_of_range_ds_raises(self):
        with self.assertRaises(ValueError):
            self.ims.compute_efficiency_ida(self.ida, ds_index=99)


# ---------------------------------------------------------------------------
# TestProficiencyMCA
# ---------------------------------------------------------------------------

class TestProficiencyMCA(unittest.TestCase):

    def setUp(self):
        self.ims = imselection()

    def test_known_lognormal_fragility(self):
        """
        For a lognormal fragility with median θ and sigma β,
        im16 = θ·exp(−β) and im84 = θ·exp(β), so proficiency = β.
        """
        from scipy.stats import norm
        theta = 0.15
        beta = 0.4
        intensities = np.geomspace(0.01, 2.0, 200)
        poes = norm.cdf(np.log(intensities / theta) / beta).reshape(-1, 1)
        cd = _make_synthetic_cloud_dict(sigma=beta)
        cd['fragility']['intensities'] = intensities
        cd['fragility']['poes'] = poes

        result = self.ims.compute_proficiency_mca(cd, ds_index=0)
        self.assertAlmostEqual(result['beta_IM_given_DCRLS1'], beta, places=2)
        self.assertEqual(result['method'], 'MCA')

    def test_nan_when_curve_flat(self):
        """When PoE never reaches 16% or 84%, proficiency is NaN."""
        cd = _make_synthetic_cloud_dict()
        intensities = np.geomspace(0.01, 2.0, 50)
        # PoE never reaches 0.84 (values all < 0.5)
        cd['fragility']['intensities'] = intensities
        cd['fragility']['poes'] = (
            np.linspace(0.0, 0.5, 50).reshape(-1, 1)
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = self.ims.compute_proficiency_mca(cd, ds_index=0)
        self.assertTrue(math.isnan(result['beta_IM_given_DCRLS1']))

    def test_beta_formula_returned(self):
        """beta_formula = sigma/b is always returned."""
        cd = _make_synthetic_cloud_dict(b1=1.2, sigma=0.4)
        result = self.ims.compute_proficiency_mca(cd, ds_index=0)
        self.assertAlmostEqual(result['beta_formula'], 0.4 / 1.2, places=10)


# ---------------------------------------------------------------------------
# TestProficiencyIDA
# ---------------------------------------------------------------------------

class TestProficiencyIDA(unittest.TestCase):

    def setUp(self):
        self.ims = imselection()

    def test_correct_beta_from_known_curves(self):
        """
        With p16 = median/exp(σ) and p84 = median*exp(σ),
        proficiency = 0.5·ln(p84/p16) = σ.
        """
        b1 = 1.2
        sigma = 0.4
        ida = _make_synthetic_ida_dict(b1=b1, sigma=sigma)
        result = self.ims.compute_proficiency_ida(ida, ds_index=0)
        # p84/p16 = exp(2σ/b1) so beta = σ/b1
        expected = sigma / b1
        self.assertAlmostEqual(result['beta_IM_given_DCRLS1'], expected, places=3)
        self.assertEqual(result['method'], 'IDA')

    def test_nan_guard_when_p16_ge_p84(self):
        """Returns NaN when p84 ≤ p16 at the damage threshold."""
        ida = _make_synthetic_ida_dict()
        edp_axis = ida['stats']['fitted_edps']
        # Force p84 = p16 everywhere
        ida['stats']['p84_im'] = ida['stats']['p16_im'].copy()
        ds_edp = ida['ida_inputs']['damage_thresholds'][0]
        # Confirm ds_edp is within range
        self.assertLessEqual(ds_edp, edp_axis[-1])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = self.ims.compute_proficiency_ida(ida, ds_index=0)
        self.assertTrue(math.isnan(result['beta_IM_given_DCRLS1']))


# ---------------------------------------------------------------------------
# TestRSMMCA
# ---------------------------------------------------------------------------

class TestRSMMCA(unittest.TestCase):

    def setUp(self):
        self.ims = imselection()

    def test_rsm_zero_when_identical_im(self):
        """RSM ≈ 0 bits when IM₂ = IM₁ (same cloud_dict passed twice)."""
        cd = _make_synthetic_cloud_dict(n=100, seed=42)
        result = self.ims.compute_rsm_mca(cd, cd)
        self.assertAlmostEqual(result['rsm'], 0.0, places=8)

    def test_rsm_positive_when_im2_more_correlated(self):
        """
        RSM(IM₂ vs IM₁) > 0 when IM₂ is more correlated with EDP.

        Setup:
        - im_true drives EDP via the power-law model.
        - IM₂ = im_true   (perfect predictor; regression sigma ≈ sigma_true).
        - IM₁ = im_true * exp(noise)  (noisy; regression sigma > sigma_true).
        Both cloud_dicts are built with OLS-fitted parameters so they are
        consistent with their respective raw data.
        """
        rng = np.random.default_rng(7)
        n = 120
        b1_true = 1.2
        b0_true = -3.5
        sigma_true = 0.25

        # True IM and EDP
        im_true = np.exp(rng.uniform(np.log(0.05), np.log(0.5), n))
        ln_edp = b0_true + b1_true * np.log(im_true) + rng.normal(0, sigma_true, n)
        edp_nc = np.exp(ln_edp)

        # IM1: noisy proxy (add i.i.d. lognormal noise, sigma_noise = 0.45)
        im1_nc = im_true * np.exp(rng.normal(0, 0.45, n))
        # IM2: the true IM (much more correlated with EDP)
        im2_nc = im_true.copy()

        def _fit_and_make_cd(im_arr):
            """OLS-fit regression on (im_arr, edp_nc) and build cloud_dict."""
            ln_im = np.log(np.maximum(im_arr, 1e-10))
            b1_fit, b0_fit = np.polyfit(ln_im, ln_edp, 1)
            resid = ln_edp - b0_fit - b1_fit * ln_im
            sigma_fit = max(np.std(resid), 1e-6)
            intensities = np.geomspace(0.01, 2.0, 50)
            from scipy.stats import norm
            poes = norm.cdf(
                np.log(intensities / 0.15) / sigma_fit).reshape(-1, 1)
            return {
                'regression': {
                    'b1': b1_fit, 'b0': b0_fit, 'sigma': sigma_fit,
                    'alpha0': -5.0, 'alpha1': 3.0,
                    'fitted_x': np.log(intensities),
                    'fitted_y': b0_fit + b1_fit * np.log(intensities),
                },
                'fragility': {
                    'intensities': intensities, 'poes': poes,
                    'medians': [0.15], 'sigma_record2record': [sigma_fit],
                },
                'bootstraps': {},
                'raw_data': {
                    'im_nc': im_arr.copy(),
                    'edp_nc': edp_nc.copy(),
                    'im_c': np.array([]),
                },
            }

        cd1 = _fit_and_make_cd(im1_nc)
        cd2 = _fit_and_make_cd(im2_nc)

        # IM₂ should explain EDP better → sigma2 < sigma1 → RSM > 0
        self.assertLess(cd2['regression']['sigma'], cd1['regression']['sigma'])
        result = self.ims.compute_rsm_mca(cd1, cd2)
        self.assertGreater(result['rsm'], 0.0)

    def test_n_records_correct(self):
        cd = _make_synthetic_cloud_dict(n=60, seed=3)
        result = self.ims.compute_rsm_mca(cd, cd)
        self.assertEqual(result['n_records'], 60)

    def test_rsm_per_record_shape(self):
        cd = _make_synthetic_cloud_dict(n=50, seed=5)
        result = self.ims.compute_rsm_mca(cd, cd)
        self.assertEqual(len(result['rsm_per_record']), 50)

    def test_mismatched_record_count_raises(self):
        cd1 = _make_synthetic_cloud_dict(n=40, seed=1)
        cd2 = _make_synthetic_cloud_dict(n=50, seed=2)
        with self.assertRaises(ValueError):
            self.ims.compute_rsm_mca(cd1, cd2)

    def test_empty_noncollapse_raises(self):
        cd = _make_synthetic_cloud_dict(n=10, seed=0)
        cd['raw_data']['im_nc'] = np.array([])
        cd['raw_data']['edp_nc'] = np.array([])
        with self.assertRaises(ValueError):
            self.ims.compute_rsm_mca(cd, cd)

    def test_collapse_heavy_does_not_crash(self):
        """Should complete (not raise) even when alpha0/alpha1 predict near-1 collapse."""
        cd = _make_synthetic_cloud_dict(
            n=80, alpha0=10.0, alpha1=0.1, seed=9
        )
        result = self.ims.compute_rsm_mca(cd, cd)
        self.assertTrue(math.isfinite(result['rsm']))


# ---------------------------------------------------------------------------
# TestRSMIDA
# ---------------------------------------------------------------------------

class TestRSMIDA(unittest.TestCase):

    def setUp(self):
        self.ims = imselection()

    def test_rsm_finite_on_synthetic_data(self):
        ida = _make_synthetic_ida_dict(n_records=30, seed=10)
        result = self.ims.compute_rsm_ida(ida, ida)
        self.assertTrue(math.isfinite(result['rsm']))
        self.assertEqual(result['method'], 'IDA')

    def test_rsm_zero_when_identical(self):
        """RSM ≈ 0 when both IDA dicts are identical."""
        ida = _make_synthetic_ida_dict(n_records=40, seed=11)
        result = self.ims.compute_rsm_ida(ida, ida)
        self.assertAlmostEqual(result['rsm'], 0.0, places=6)

    def test_n_valid_le_n_records(self):
        ida = _make_synthetic_ida_dict(n_records=25, seed=12)
        result = self.ims.compute_rsm_ida(ida, ida)
        self.assertLessEqual(result['n_valid'], result['n_records'])

    def test_mismatched_record_count_raises(self):
        ida1 = _make_synthetic_ida_dict(n_records=20, seed=1)
        ida2 = _make_synthetic_ida_dict(n_records=30, seed=2)
        with self.assertRaises(ValueError):
            self.ims.compute_rsm_ida(ida1, ida2)


# ---------------------------------------------------------------------------
# TestRSMGeneral
# ---------------------------------------------------------------------------

class TestRSMGeneral(unittest.TestCase):

    def setUp(self):
        self.ims = imselection()

    def test_log2_c_when_pdf2_equals_c_times_pdf1(self):
        """When f2 = c·f1 for all records, RSM = log2(c)."""
        c = 3.0
        demands = np.ones(20) * 0.01
        im1 = np.ones(20) * 0.1
        im2 = np.ones(20) * 0.1
        f1 = lambda d, im: 1.0  # noqa: E731
        f2 = lambda d, im: c    # noqa: E731
        result = self.ims.compute_rsm_general(demands, im1, im2, f1, f2)
        self.assertAlmostEqual(result['rsm'], math.log2(c), places=8)

    def test_rsm_zero_when_pdfs_equal(self):
        """RSM = 0 when both PDF callables return the same value."""
        f = lambda d, im: 0.5  # noqa: E731
        demands = np.linspace(0.001, 0.05, 30)
        im = np.linspace(0.1, 0.5, 30)
        result = self.ims.compute_rsm_general(demands, im, im, f, f)
        self.assertAlmostEqual(result['rsm'], 0.0, places=10)

    def test_no_crash_on_zero_pdf(self):
        """Zero PDF is floored, should not crash or produce -inf."""
        f1 = lambda d, im: 0.0  # noqa: E731
        f2 = lambda d, im: 1e-5  # noqa: E731
        demands = np.ones(10) * 0.01
        im = np.ones(10) * 0.1
        result = self.ims.compute_rsm_general(demands, im, im, f1, f2)
        self.assertTrue(math.isfinite(result['rsm']))

    def test_method_label(self):
        f = lambda d, im: 1.0  # noqa: E731
        result = self.ims.compute_rsm_general(
            np.ones(5), np.ones(5), np.ones(5), f, f
        )
        self.assertEqual(result['method'], 'general')


# ---------------------------------------------------------------------------
# TestCompareIMs
# ---------------------------------------------------------------------------

class TestCompareIMs(unittest.TestCase):

    def setUp(self):
        self.ims = imselection()

    def _three_cloud_dicts(self):
        return {
            'PGA': _make_synthetic_cloud_dict(b1=1.0, sigma=0.55, n=60, seed=0),
            'Sa(T1)': _make_synthetic_cloud_dict(b1=1.2, sigma=0.40, n=60, seed=1),
            'Saavg': _make_synthetic_cloud_dict(b1=1.3, sigma=0.30, n=60, seed=2),
        }

    def _three_ida_dicts(self):
        return {
            'PGA': _make_synthetic_ida_dict(n_records=20, sigma=0.55, seed=0),
            'Sa(T1)': _make_synthetic_ida_dict(n_records=20, sigma=0.40, seed=1),
            'Saavg': _make_synthetic_ida_dict(n_records=20, sigma=0.30, seed=2),
        }

    def test_dataframe_has_expected_columns(self):
        result = self.ims.compare_ims(
            self._three_cloud_dicts(), analysis_type='MCA'
        )
        df = result['ranking']
        for col in ('im_name', 'efficiency', 'proficiency',
                    'rsm_vs_reference', 'rank_efficiency',
                    'rank_proficiency', 'rank_rsm'):
            self.assertIn(col, df.columns, msg=f"Missing column: {col}")

    def test_ranks_are_integers_1_to_n(self):
        result = self.ims.compare_ims(
            self._three_cloud_dicts(), analysis_type='MCA'
        )
        df = result['ranking']
        n = len(df)
        for col in ('rank_efficiency', 'rank_proficiency', 'rank_rsm'):
            ranks = sorted(df[col].tolist())
            self.assertEqual(ranks, list(range(1, n + 1)),
                             msg=f"Ranks in {col} are not 1..{n}: {ranks}")

    def test_single_im_raises(self):
        with self.assertRaises(ValueError):
            self.ims.compare_ims(
                {'PGA': _make_synthetic_cloud_dict()}, analysis_type='MCA'
            )

    def test_invalid_analysis_type_raises(self):
        with self.assertRaises(ValueError):
            self.ims.compare_ims(
                self._three_cloud_dicts(), analysis_type='CLOUD'
            )

    def test_ida_path_exercised(self):
        result = self.ims.compare_ims(
            self._three_ida_dicts(), analysis_type='IDA'
        )
        self.assertEqual(result['analysis_type'], 'IDA')
        df = result['ranking']
        self.assertEqual(len(df), 3)

    def test_rsm_matrix_keys(self):
        im_dict = self._three_cloud_dicts()
        result = self.ims.compare_ims(im_dict, analysis_type='MCA')
        mat = result['rsm_matrix']
        for k in im_dict:
            self.assertIn(k, mat)
            self.assertEqual(set(mat[k].keys()), set(im_dict.keys()))


if __name__ == '__main__':
    unittest.main()
