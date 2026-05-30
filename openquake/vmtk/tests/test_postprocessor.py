import os
import unittest
import numpy as np
import pandas as pd

from openquake.vmtk.postprocessor import postprocessor


# ---------------------------------------------------------------------------
# Shared test data helpers
# ---------------------------------------------------------------------------
def _cap_array():
    return np.array([
        [0.0,        0.0],
        [0.0005313,  1.025],
        [0.004,      2.05],
        [0.024,      2.071]])


def _damage_thresholds(cap):
    return [
        0.75 * cap[2, 0],
        0.5 * cap[2, 0] + 0.33 * cap[-1, 0],
        0.25 * cap[2, 0] + 0.67 * cap[-1, 0],
        cap[-1, 0]]


def _load_cloud_data():
    cd = os.path.dirname(__file__)
    imls = np.loadtxt(
        os.path.join(cd, 'test_data', 'imls.csv'),
        delimiter=',', usecols=0).tolist()
    edps = np.loadtxt(
        os.path.join(cd, 'test_data', 'edps.csv'),
        delimiter=',', usecols=0).tolist()
    return imls, edps


# ---------------------------------------------------------------------------
# calculate_lognormal_fragility
# ---------------------------------------------------------------------------
class TestLognormalFragility(unittest.TestCase):

    def setUp(self):
        self.pp = postprocessor()
        self.theta = 0.50
        self.sigma_rr = 0.40
        self.ims = np.round(np.geomspace(0.05, 10.0, 50), 3)

    def test_output_length_matches_intensities(self):
        poes = self.pp.calculate_lognormal_fragility(
            self.theta, self.sigma_rr,
            sigma_build2build=0.0, sigma_ds=0.0,
            intensities=np.linspace(0.1, 2.0, 30))
        self.assertEqual(len(poes), 30)

    def test_poe_at_median_equals_half(self):
        """P(exceed | IM = theta) = 0.50 by definition."""
        poe = self.pp.calculate_lognormal_fragility(
            self.theta, self.sigma_rr,
            sigma_build2build=0.0, sigma_ds=0.0,
            intensities=[self.theta])
        self.assertAlmostEqual(float(poe.item()), 0.50, places=6)

    def test_poe_at_median_with_combined_uncertainty(self):
        """PoE at IM = theta is 0.50 regardless of sigma components."""
        poe = self.pp.calculate_lognormal_fragility(
            self.theta, self.sigma_rr,
            sigma_build2build=0.30, sigma_ds=0.30,
            intensities=[self.theta])
        self.assertAlmostEqual(float(poe.item()), 0.50, places=6)

    def test_poes_bounded(self):
        poes = self.pp.calculate_lognormal_fragility(
            self.theta, self.sigma_rr,
            intensities=self.ims)
        self.assertTrue(np.all(poes >= 0.0))
        self.assertTrue(np.all(poes <= 1.0))

    def test_poes_monotonically_increasing(self):
        poes = self.pp.calculate_lognormal_fragility(
            self.theta, self.sigma_rr,
            intensities=self.ims)
        self.assertTrue(np.all(np.diff(poes) >= 0))

    def test_larger_uncertainty_flatter_curve(self):
        """Higher beta → lower PoE above the median."""
        im = np.array([self.theta * 2.0])
        poe_low = self.pp.calculate_lognormal_fragility(
            self.theta, 0.1,
            sigma_build2build=0.0, sigma_ds=0.0, intensities=im)
        poe_high = self.pp.calculate_lognormal_fragility(
            self.theta, 0.8,
            sigma_build2build=0.0, sigma_ds=0.0, intensities=im)
        self.assertGreater(float(poe_low.item()), float(poe_high.item()))


# ---------------------------------------------------------------------------
# calculate_rotated_fragility
# ---------------------------------------------------------------------------
class TestRotatedFragility(unittest.TestCase):

    def setUp(self):
        self.pp = postprocessor()
        self.theta = 0.50
        self.sigma_rr = 0.40
        self.percentile = 0.10
        self.ims = np.linspace(0.01, 2.0, 500)

    def test_returns_three_values(self):
        result = self.pp.calculate_rotated_fragility(
            self.percentile, self.theta, self.sigma_rr,
            sigma_build2build=0.0, sigma_ds=0.0,
            intensities=self.ims)
        self.assertEqual(len(result), 3)

    def test_no_rotation_when_no_epistemic(self):
        """With sigma_b2b=0 and sigma_ds=0, theta_prime = theta."""
        theta_prime, _, _ = self.pp.calculate_rotated_fragility(
            self.percentile, self.theta, self.sigma_rr,
            sigma_build2build=0.0, sigma_ds=0.0,
            intensities=self.ims)
        self.assertAlmostEqual(theta_prime, self.theta, places=6)

    def test_anchor_percentile_preserved(self):
        """IM at target percentile is same for original and rotated."""
        non_rotated = self.pp.calculate_lognormal_fragility(
            self.theta, self.sigma_rr,
            sigma_build2build=0.0, sigma_ds=0.0,
            intensities=self.ims)
        _, _, rotated = self.pp.calculate_rotated_fragility(
            self.percentile, self.theta, self.sigma_rr,
            sigma_build2build=0.30, sigma_ds=0.0,
            intensities=self.ims)
        im_nr = np.interp(self.percentile, non_rotated, self.ims)
        im_r = np.interp(self.percentile, rotated, self.ims)
        self.assertAlmostEqual(im_nr, im_r, places=3)

    def test_poes_bounded_and_monotonic(self):
        _, _, poes = self.pp.calculate_rotated_fragility(
            self.percentile, self.theta, self.sigma_rr,
            intensities=self.ims)
        self.assertTrue(np.all(poes >= 0.0))
        self.assertTrue(np.all(poes <= 1.0))
        self.assertTrue(np.all(np.diff(poes) >= 0))

    def test_output_length_matches_intensities(self):
        _, _, poes = self.pp.calculate_rotated_fragility(
            self.percentile, self.theta, self.sigma_rr,
            intensities=self.ims)
        self.assertEqual(len(poes), len(self.ims))


# ---------------------------------------------------------------------------
# calculate_glm_fragility
# ---------------------------------------------------------------------------
class TestGLMFragility(unittest.TestCase):

    def setUp(self):
        self.pp = postprocessor()
        cap = _cap_array()
        self.imls, self.edps = _load_cloud_data()
        self.thresholds = _damage_thresholds(cap)
        self.ims = np.round(np.geomspace(0.05, 10.0, 50), 3)

    def test_logit_output_shape(self):
        poes = self.pp.calculate_glm_fragility(
            self.imls, self.edps, self.thresholds,
            intensities=self.ims,
            fragility_method='logit')
        self.assertEqual(
            poes.shape,
            (len(self.ims), len(self.thresholds)))

    def test_probit_output_shape(self):
        poes = self.pp.calculate_glm_fragility(
            self.imls, self.edps, self.thresholds,
            intensities=self.ims,
            fragility_method='probit')
        self.assertEqual(
            poes.shape,
            (len(self.ims), len(self.thresholds)))

    def test_logit_poes_bounded(self):
        poes = self.pp.calculate_glm_fragility(
            self.imls, self.edps, self.thresholds,
            fragility_method='logit')
        self.assertTrue(np.all(poes >= 0.0))
        self.assertTrue(np.all(poes <= 1.0))

    def test_logit_curves_monotonically_increasing(self):
        poes = self.pp.calculate_glm_fragility(
            self.imls, self.edps, self.thresholds,
            fragility_method='logit')
        for ds in range(poes.shape[1]):
            self.assertTrue(
                np.all(np.diff(poes[:, ds]) >= -1e-9),
                f"Column {ds} is not monotonically increasing")


# ---------------------------------------------------------------------------
# calculate_ordinal_fragility
# ---------------------------------------------------------------------------
class TestOrdinalFragility(unittest.TestCase):

    def setUp(self):
        self.pp = postprocessor()
        cap = _cap_array()
        self.imls, self.edps = _load_cloud_data()
        self.thresholds = _damage_thresholds(cap)
        self.ims = np.round(np.geomspace(0.05, 10.0, 50), 3)

    def test_output_shape_includes_extra_ds_column(self):
        """Shape must be (n_IM, n_DS + 1)."""
        poes = self.pp.calculate_ordinal_fragility(
            self.imls, self.edps, self.thresholds,
            intensities=self.ims)
        self.assertEqual(
            poes.shape,
            (len(self.ims), len(self.thresholds) + 1))

    def test_poes_bounded(self):
        poes = self.pp.calculate_ordinal_fragility(
            self.imls, self.edps, self.thresholds,
            intensities=self.ims)
        self.assertTrue(np.all(poes >= -1e-9))
        self.assertTrue(np.all(poes <= 1.0 + 1e-9))


# ---------------------------------------------------------------------------
# process_mca_results
# ---------------------------------------------------------------------------
class TestModifiedCloudAnalysis(unittest.TestCase):
    """
    Regression coefficients are compared against the ESRM20 outputs
    for building class 'CR_LDUAL-DUH_H1', PGA.
    Reference:
    https://gitlab.seismo.ethz.ch/efehr/esrm20_vulnerability/
    """

    B0_EXPECTED = -6.657090
    B1_EXPECTED = 2.491236
    SIGMA_EXPECTED = 0.493070
    TOLERANCE = 0.01  # 1 % relative tolerance

    def setUp(self):
        self.pp = postprocessor()
        cap = _cap_array()
        self.imls, self.edps = _load_cloud_data()
        self.thresholds = _damage_thresholds(cap)
        self.lower_limit = 0.1 * cap[2, 0]
        self.censored_limit = 1.5 * cap[-1, 0]

    def _run(self, method='lognormal', **kw):
        return self.pp.process_mca_results(
            self.imls, self.edps, self.thresholds,
            self.lower_limit, self.censored_limit,
            sigma_build2build=0.30,
            fragility_method=method,
            random_seed=42,
            **kw)

    # --- dict structure ---------------------------------------------------

    def test_lognormal_top_level_keys(self):
        result = self._run()
        for key in ['cloud inputs', 'fragility',
                    'regression', 'bootstraps', 'raw_data']:
            self.assertIn(key, result)

    def test_cloud_inputs_keys(self):
        result = self._run()
        for key in ['imls', 'edps', 'lower_limit',
                    'upper_limit', 'damage_thresholds']:
            self.assertIn(key, result['cloud inputs'])

    def test_fragility_keys(self):
        result = self._run()
        for key in ['fragility_method', 'intensities', 'poes',
                    'medians', 'sigma_record2record',
                    'sigma_build2build', 'sigma_ds', 'betas_total']:
            self.assertIn(key, result['fragility'])

    def test_regression_keys(self):
        result = self._run()
        for key in ['b0', 'b1', 'sigma', 'fitted_x', 'fitted_y']:
            self.assertIn(key, result['regression'])

    def test_bootstraps_keys(self):
        result = self._run()
        for key in ['b1', 'a', 'sigma_rr',
                    'alpha0', 'alpha1', 'poes_all']:
            self.assertIn(key, result['bootstraps'])

    def test_raw_data_keys(self):
        result = self._run()
        for key in ['im_nc', 'edp_nc', 'im_c']:
            self.assertIn(key, result['raw_data'])

    # --- regression values ------------------------------------------------

    def test_regression_b0(self):
        result = self._run()
        v = result['regression']['b0']
        self.assertTrue(
            np.isclose(v, self.B0_EXPECTED, rtol=self.TOLERANCE),
            f"b0={v:.4f}, expected {self.B0_EXPECTED}")

    def test_regression_b1(self):
        result = self._run()
        v = result['regression']['b1']
        self.assertTrue(
            np.isclose(v, self.B1_EXPECTED, rtol=self.TOLERANCE),
            f"b1={v:.4f}, expected {self.B1_EXPECTED}")

    def test_regression_sigma(self):
        result = self._run()
        v = result['regression']['sigma']
        self.assertTrue(
            np.isclose(v, self.SIGMA_EXPECTED, rtol=self.TOLERANCE),
            f"sigma={v:.4f}, expected {self.SIGMA_EXPECTED}")

    # --- fragility output -------------------------------------------------

    def test_poes_shape(self):
        result = self._run()
        poes = result['fragility']['poes']
        # lognormal method appends a collapse column → n_DS + 1
        self.assertEqual(poes.shape[1], len(self.thresholds) + 1)

    def test_medians_count(self):
        result = self._run()
        self.assertEqual(
            len(result['fragility']['medians']),
            len(self.thresholds))

    def test_medians_are_positive(self):
        result = self._run()
        self.assertTrue(
            all(m > 0 for m in result['fragility']['medians']))

    def test_poes_bounded(self):
        result = self._run()
        poes = result['fragility']['poes']
        self.assertTrue(np.all(poes >= 0.0))
        self.assertTrue(np.all(poes <= 1.0))

    # --- alternative fragility methods ------------------------------------

    def test_logit_method_accepted(self):
        result = self._run(method='logit')
        self.assertEqual(
            result['fragility']['fragility_method'], 'logit')

    def test_ordinal_method_accepted(self):
        result = self._run(method='ordinal')
        self.assertEqual(
            result['fragility']['fragility_method'], 'ordinal')

    def test_fragility_rotation_accepted(self):
        result = self._run(
            fragility_rotation=True, rotation_percentile=0.10)
        self.assertIn('fragility', result)

    def test_cloud_method_key_bootstrap(self):
        result = self._run(cloud_method='bootstrap')
        self.assertEqual(
            result['fragility']['cloud_method'], 'bootstrap')

    def test_cloud_method_key_classical(self):
        result = self._run(
            cloud_method='classical',
            n_mcmc=300, n_mcmc_burnin=100)
        self.assertEqual(
            result['fragility']['cloud_method'], 'classical')

    def test_classical_top_level_keys(self):
        result = self._run(
            cloud_method='classical',
            n_mcmc=300, n_mcmc_burnin=100)
        for key in ['cloud inputs', 'fragility',
                    'regression', 'mcmc', 'raw_data']:
            self.assertIn(key, result)

    def test_classical_fragility_extra_keys(self):
        result = self._run(
            cloud_method='classical',
            n_mcmc=300, n_mcmc_burnin=100)
        frag = result['fragility']
        for key in ['poes_robust_plus', 'poes_robust_minus',
                    'confidence_k']:
            self.assertIn(key, frag)

    def test_classical_poes_shape(self):
        result = self._run(
            cloud_method='classical',
            n_mcmc=300, n_mcmc_burnin=100)
        poes = result['fragility']['poes']
        self.assertEqual(poes.shape[1], len(self.thresholds) + 1)

    def test_classical_poes_bounded(self):
        result = self._run(
            cloud_method='classical',
            n_mcmc=300, n_mcmc_burnin=100)
        frag = result['fragility']
        for key in ['poes', 'poes_robust_plus', 'poes_robust_minus']:
            self.assertTrue(np.all(frag[key] >= 0.0))
            self.assertTrue(np.all(frag[key] <= 1.0))

    def test_classical_mcmc_keys(self):
        result = self._run(
            cloud_method='classical',
            n_mcmc=300, n_mcmc_burnin=100)
        mc = result['mcmc']
        for key in ['chain', 'acceptance_rate', 'n_mcmc',
                    'n_burnin', 'chi_labels', 'chi_mean',
                    'chi_std', 'chi_median', 'poes_all']:
            self.assertIn(key, mc)

    def test_classical_mcmc_chain_shape(self):
        n_mcmc, n_burnin = 300, 100
        result = self._run(
            cloud_method='classical',
            n_mcmc=n_mcmc, n_mcmc_burnin=n_burnin)
        chain = result['mcmc']['chain']
        self.assertEqual(chain.shape, (n_mcmc - n_burnin, 5))

    def test_classical_acceptance_rate_in_range(self):
        result = self._run(
            cloud_method='classical',
            n_mcmc=300, n_mcmc_burnin=100)
        rate = result['mcmc']['acceptance_rate']
        self.assertGreater(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_unknown_cloud_method_raises(self):
        with self.assertRaises((ValueError, Exception)):
            self._run(cloud_method='invalid')


# ---------------------------------------------------------------------------
# process_msa_results
# ---------------------------------------------------------------------------
class TestMultipleStripeAnalysis(unittest.TestCase):

    def setUp(self):
        self.pp = postprocessor()
        rng = np.random.default_rng(0)
        stripe_imls = np.array([0.1, 0.5, 1.0])
        n_rec = 30
        self.imls = np.tile(stripe_imls, (n_rec, 1))
        self.edps = np.column_stack([
            rng.lognormal(np.log(0.002), 0.5, n_rec),
            rng.lognormal(np.log(0.010), 0.5, n_rec),
            rng.lognormal(np.log(0.030), 0.5, n_rec),
        ])
        self.thresholds = [0.005, 0.015, 0.030]
        self.ims = np.round(np.geomspace(0.05, 5.0, 40), 3)

    def _run(self, **kw):
        return self.pp.process_msa_results(
            self.imls, self.edps, self.thresholds,
            intensities=self.ims, **kw)

    def test_top_level_keys(self):
        for key in ['msa_inputs', 'fragility', 'metadata']:
            self.assertIn(key, self._run())

    def test_msa_inputs_keys(self):
        result = self._run()
        for key in ['imls', 'edps', 'damage_thresholds',
                    'sigma_build2build', 'sigma_ds', 'is_rotated']:
            self.assertIn(key, result['msa_inputs'])

    def test_fragility_keys(self):
        result = self._run()
        for key in ['fragility_method', 'intensities', 'poes',
                    'medians', 'sigma_record2record', 'betas_total']:
            self.assertIn(key, result['fragility'])

    def test_metadata_keys(self):
        result = self._run()
        for key in ['stripe_levels', 'observed_fractions']:
            self.assertIn(key, result['metadata'])

    def test_poes_shape(self):
        poes = self._run()['fragility']['poes']
        self.assertEqual(
            poes.shape,
            (len(self.ims), len(self.thresholds)))

    def test_medians_count(self):
        self.assertEqual(
            len(self._run()['fragility']['medians']),
            len(self.thresholds))

    def test_poes_bounded(self):
        poes = self._run()['fragility']['poes']
        self.assertTrue(np.all(poes >= 0.0))
        self.assertTrue(np.all(poes <= 1.0))

    def test_observed_fractions_shape(self):
        fracs = self._run()['metadata']['observed_fractions']
        # one list per damage state, one entry per stripe
        self.assertEqual(len(fracs), len(self.thresholds))
        self.assertEqual(len(fracs[0]), self.imls.shape[1])

    def test_fragility_rotation_flag_stored(self):
        result = self._run(
            fragility_rotation=True, rotation_percentile=0.10)
        self.assertTrue(result['msa_inputs']['is_rotated'])


# ---------------------------------------------------------------------------
# calculate_vulnerability_function
# ---------------------------------------------------------------------------
class TestVulnerabilityFunction(unittest.TestCase):

    def setUp(self):
        self.pp = postprocessor()
        self.ims = np.round(np.geomspace(0.05, 10.0, 50), 3)
        poes_1d = self.pp.calculate_lognormal_fragility(
            0.50, 0.40,
            sigma_build2build=0.0, sigma_ds=0.0,
            intensities=self.ims)
        self.poes = poes_1d[:, np.newaxis]  # shape (50, 1)
        self.consequence_model = [1.0]

    def _vuln(self, **kw):
        return self.pp.calculate_vulnerability_function(
            self.poes, self.consequence_model,
            uncertainty=False,
            intensities=self.ims, **kw)

    def test_returns_dataframe(self):
        self.assertIsInstance(self._vuln(), pd.DataFrame)

    def test_dataframe_columns(self):
        df = self._vuln()
        for col in ['IML', 'Loss', 'COV']:
            self.assertIn(col, df.columns)

    def test_iml_column_matches_intensities(self):
        df = self._vuln()
        np.testing.assert_array_equal(df['IML'].values, self.ims)

    def test_loss_bounded(self):
        df = self._vuln()
        self.assertTrue(np.all(df['Loss'].values >= 0.0))
        self.assertTrue(np.all(df['Loss'].values <= 1.0))

    def test_loss_monotonically_increasing(self):
        df = self._vuln()
        self.assertTrue(np.all(np.diff(df['Loss'].values) >= -1e-9))

    def test_loss_at_high_im_matches_poe(self):
        """At the highest IM, Loss ≈ PoE for a single DS with LR=1."""
        df = self._vuln()
        self.assertAlmostEqual(
            df['Loss'].values[-1],
            float(self.poes[-1].item()),
            places=2)

    def test_no_uncertainty_zero_cov(self):
        df = self._vuln()
        self.assertTrue(np.all(df['COV'].values == 0.0))

    def test_silva_uncertainty_accepted(self):
        df = self.pp.calculate_vulnerability_function(
            self.poes, self.consequence_model,
            uncertainty=True, method='silva',
            intensities=self.ims)
        self.assertIn('COV', df.columns)
        self.assertTrue(np.all(df['COV'].values >= 0.0))


# ---------------------------------------------------------------------------
# calculate_average_annual_damage_probability /
# calculate_average_annual_loss
# ---------------------------------------------------------------------------
class TestAverageAnnualMetrics(unittest.TestCase):

    def setUp(self):
        self.pp = postprocessor()
        im_vals = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
        rates = np.array([0.10, 0.05, 0.02, 0.008, 0.002, 0.0005, 0.0001])
        self.hazard = np.column_stack([im_vals, rates])
        poes = self.pp.calculate_lognormal_fragility(
            0.5, 0.4,
            sigma_build2build=0.0, sigma_ds=0.0,
            intensities=im_vals)
        self.fragility = np.column_stack([im_vals, poes])
        self.vulnerability = np.column_stack([im_vals, poes])

    def test_aadp_returns_scalar(self):
        result = self.pp.calculate_average_annual_damage_probability(
            self.fragility, self.hazard)
        self.assertIsInstance(float(result), float)

    def test_aadp_bounded(self):
        result = self.pp.calculate_average_annual_damage_probability(
            self.fragility, self.hazard)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_aal_returns_scalar(self):
        result = self.pp.calculate_average_annual_loss(
            self.vulnerability, self.hazard)
        self.assertIsInstance(float(result), float)

    def test_aal_bounded(self):
        result = self.pp.calculate_average_annual_loss(
            self.vulnerability, self.hazard)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_aadp_zero_when_hazard_below_threshold(self):
        """All hazard rates below the filter → AADP = 0."""
        low_hazard = np.array([
            [0.05, 0.0001],
            [0.10, 0.00005]])
        result = self.pp.calculate_average_annual_damage_probability(
            self.fragility, low_hazard,
            max_return_period=100)
        self.assertEqual(result, 0.0)

    def test_aal_zero_when_hazard_below_threshold(self):
        """All hazard rates below the filter → AAL = 0."""
        low_hazard = np.array([
            [0.05, 0.0001],
            [0.10, 0.00005]])
        result = self.pp.calculate_average_annual_loss(
            self.vulnerability, low_hazard,
            max_return_period=100)
        self.assertEqual(result, 0.0)


if __name__ == '__main__':
    unittest.main()
