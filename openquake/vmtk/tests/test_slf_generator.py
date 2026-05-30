import os
import unittest
import numpy as np
import pandas as pd

from openquake.vmtk.slfgenerator import slfgenerator


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

def _load_inventory(edp_type):
    """Return the test inventory filtered to the requested EDP type."""
    cd = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(cd, "test_data", "slf_inventory.csv"))
    return df[df["EDP"].str.lower() == edp_type.lower()]


def _make_model(edp="PSD", **overrides):
    """Construct a :class:`slfgenerator` with sensible test defaults."""
    cd = os.path.dirname(__file__)
    slf_file = pd.read_csv(os.path.join(cd, "test_data", "slf_inventory.csv"))
    kwargs = dict(
        component_data=slf_file,
        edp=edp,
        grouping_flag=True,
        conversion=1.0,
        realizations=100,
        replacement_cost=1.0,
        regression="gpd",
    )
    kwargs.update(overrides)
    return slfgenerator(**kwargs)


# ---------------------------------------------------------------------------
# Initialisation and validation tests
# ---------------------------------------------------------------------------

class TestSLFGeneratorInit(unittest.TestCase):

    def test_valid_psd_inputs_accepted(self):
        model = _make_model(edp="PSD")
        self.assertEqual(model.edp, "psd")
        self.assertTrue(model.grouping_flag)

    def test_valid_pfa_inputs_accepted(self):
        model = _make_model(edp="PFA", regression="weibull")
        self.assertEqual(model.edp, "pfa")

    def test_invalid_edp_raises(self):
        with self.assertRaises(ValueError):
            _make_model(edp="SAV")

    def test_zero_replacement_cost_raises(self):
        model = _make_model()
        fragilities, means_cost, covs_cost = model.fragility_function()
        damage_state = model.do_monte_carlo_simulations(fragilities)
        damage_state = model.validate_ds_dependence(damage_state)
        group_data = next(iter(model.component_groups.values()))
        item_ids = list(group_data["Component ID"])
        ds_group = {k: damage_state[k] for k in item_ids}
        model.replacement_cost = 0.0
        with self.assertRaises(ValueError):
            model.calculate_costs(ds_group, means_cost, covs_cost)

    def test_edp_range_default_psd(self):
        model = _make_model(edp="PSD")
        self.assertGreater(len(model.edp_range), 1)
        self.assertAlmostEqual(model.edp_range[0], 1e-20)

    def test_custom_edp_range_accepted(self):
        custom_range = np.linspace(0, 0.1, 50)
        model = _make_model(edp="PSD", edp_range=custom_range)
        self.assertEqual(len(model.edp_range), 50)

    def test_regression_normalised_to_lowercase(self):
        model = _make_model(regression="Weibull")
        self.assertEqual(model.regression, "weibull")

    def test_no_regression_accepts_none(self):
        model = _make_model(regression=None)
        self.assertIsNone(model.regression)

    def test_component_groups_populated(self):
        model = _make_model()
        self.assertGreater(len(model.component_groups), 0)


# ---------------------------------------------------------------------------
# Core pipeline tests
# ---------------------------------------------------------------------------

class TestSLFGeneratorPipeline(unittest.TestCase):

    def setUp(self):
        self.model = _make_model()

    def test_fragility_function_returns_three_items(self):
        result = self.model.fragility_function()
        self.assertEqual(len(result), 3)

    def test_fragility_edp_key_present(self):
        fragilities, _, _ = self.model.fragility_function()
        self.assertIn("EDP", fragilities)
        self.assertIn("IDs", fragilities)

    def test_fragility_curves_bounded(self):
        fragilities, _, _ = self.model.fragility_function()
        for item_frags in fragilities["IDs"].values():
            for curve in item_frags.values():
                self.assertTrue(np.all(curve >= 0.0))
                self.assertTrue(np.all(curve <= 1.0))

    def test_monte_carlo_returns_dict(self):
        fragilities, _, _ = self.model.fragility_function()
        ds = self.model.do_monte_carlo_simulations(fragilities)
        self.assertIsInstance(ds, dict)
        self.assertEqual(len(ds), len(fragilities["IDs"]))

    def test_monte_carlo_realization_count(self):
        fragilities, _, _ = self.model.fragility_function()
        ds = self.model.do_monte_carlo_simulations(fragilities)
        first_item = next(iter(ds.values()))
        self.assertEqual(len(first_item), self.model.realizations)

    def test_validate_ds_dependence_no_tree(self):
        """Without a correlation tree the damage states are unchanged."""
        fragilities, _, _ = self.model.fragility_function()
        ds_before = self.model.do_monte_carlo_simulations(fragilities)
        ds_after = self.model.validate_ds_dependence(ds_before)
        self.assertIs(ds_before, ds_after)

    def test_calculate_costs_returns_three_items(self):
        fragilities, means_cost, covs_cost = self.model.fragility_function()
        ds = self.model.do_monte_carlo_simulations(fragilities)
        ds = self.model.validate_ds_dependence(ds)
        group_data = next(iter(self.model.component_groups.values()))
        item_ids = list(group_data["Component ID"])
        ds_group = {k: ds[k] for k in item_ids}
        result = self.model.calculate_costs(ds_group, means_cost, covs_cost)
        self.assertEqual(len(result), 3)

    def test_total_loss_storey_length(self):
        fragilities, means_cost, covs_cost = self.model.fragility_function()
        ds = self.model.do_monte_carlo_simulations(fragilities)
        ds = self.model.validate_ds_dependence(ds)
        group_data = next(iter(self.model.component_groups.values()))
        item_ids = list(group_data["Component ID"])
        ds_group = {k: ds[k] for k in item_ids}
        total, _, _ = self.model.calculate_costs(
            ds_group, means_cost, covs_cost
        )
        self.assertEqual(len(total), self.model.realizations)

    def test_estimate_accuracy_returns_two_floats(self):
        y = np.linspace(0, 1, 100)
        yhat = y + 0.01
        em, ec = self.model.estimate_accuracy(y, yhat)
        self.assertIsInstance(float(em), float)
        self.assertIsInstance(float(ec), float)

    def test_estimate_accuracy_perfect_fit_is_zero(self):
        y = np.linspace(0.01, 1, 100)
        em, ec = self.model.estimate_accuracy(y, y)
        self.assertAlmostEqual(em, 0.0, places=10)
        self.assertAlmostEqual(ec, 0.0, places=10)


# ---------------------------------------------------------------------------
# generate() end-to-end tests
# ---------------------------------------------------------------------------

class TestSLFGeneratorGenerate(unittest.TestCase):

    def setUp(self):
        self.model = _make_model()
        self.out, self.cache = self.model.generate()

    def test_generate_returns_two_items(self):
        result = self.model.generate()
        self.assertEqual(len(result), 2)

    def test_out_is_dict(self):
        self.assertIsInstance(self.out, dict)

    def test_cache_is_dict(self):
        self.assertIsInstance(self.cache, dict)

    def test_out_keys_match_cache_keys(self):
        self.assertEqual(set(self.out), set(self.cache))

    def test_out_has_required_keys(self):
        for group_out in self.out.values():
            for key in ("edp", "edp_range", "slf", "error_max", "error_cum"):
                self.assertIn(key, group_out)

    def test_cache_has_required_keys(self):
        for group_cache in self.cache.values():
            for key in (
                "component", "fragilities", "total_loss_storey",
                "total_loss_storey_ratio", "losses", "slfs",
                "empirical_median", "empirical_16th", "empirical_84th",
            ):
                self.assertIn(key, group_cache)

    def test_slf_length_matches_edp_range(self):
        for group_out in self.out.values():
            self.assertEqual(
                len(group_out["slf"]), len(group_out["edp_range"])
            )

    def test_empirical_median_shape(self):
        for group_cache in self.cache.values():
            median = group_cache["empirical_median"]
            self.assertEqual(len(median), len(self.model.edp_range))

    def test_error_max_is_non_negative(self):
        for group_out in self.out.values():
            self.assertGreaterEqual(group_out["error_max"], 0.0)

    def test_regression_stored_in_cache(self):
        for group_cache in self.cache.values():
            self.assertIsNotNone(group_cache["regression"])


if __name__ == "__main__":
    unittest.main()
