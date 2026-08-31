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
    )
    kwargs.update(overrides)
    return slfgenerator(**kwargs)


def _correlation_tree_fixture():
    """Correlation tree for the 3-component test fixture (slf_inventory.csv).

    Component 2 (EBCJ) depends on component 1 (IBCJ) reaching DS2 -- once
    component 1 reaches DS2, DS3, or DS4, component 2 is forced to at
    least DS1. Component 3 (Column) is independent. Max damage state
    across all three components is 4 (components 1 and 2), so the tree
    needs DS0..DS4 columns.
    """
    return pd.DataFrame({
        "ID":                [1, 2, 3],
        "DEPENDENT ON ITEM": ["Independent", "1", "Independent"],
        "DS0":               ["Independent", "Independent", "Independent"],
        "DS1":               ["Independent", "Independent", "Independent"],
        "DS2":               ["Independent", "DS1",          "Independent"],
        "DS3":               ["Independent", "DS1",          "Independent"],
        "DS4":               ["Independent", "DS1",          "Independent"],
    })


def _non_sequential_id_inventory():
    """A 3-component inventory with non-sequential, non-contiguous
    Component IDs (25, 10, 3) in scrambled row order, used to prove
    slfgenerator's correlation-tree join is genuinely ID-based rather
    than positional (row-order) based.
    """
    def row(cid, n_ds, medians, disps, costs, cost_disps):
        r = {"Component ID": cid, "Description": "B", "EDP": "PSD",
             "Typology": "S", "Performance Group": 1, "Quantity": 1,
             "Damage States": n_ds}
        for i in range(4):
            in_range = i < n_ds
            r[f"DS{i + 1}, Median"] = medians[i] if in_range else np.nan
            r[f"DS{i + 1}, Total Dispersion"] = disps[i] if in_range else np.nan
            r[f"DS{i + 1}, Cost"] = costs[i] if in_range else np.nan
            r[f"DS{i + 1}, Cost Dispersion"] = cost_disps[i] if in_range else np.nan
            r[f"DS{i + 1}, Best Fit"] = "Lognormal" if in_range else np.nan
        return r

    return pd.DataFrame([
        row(25, 4, [0.0074, 0.025, 0.046, 0.056], [0.3, 0.24, 0.25, 0.25],
            [0.71, 1.68, 3.09, 4.23], [1.78, 1.5, 2.12, 2.35]),
        row(10, 4, [0.0057, 0.021, 0.037, 0.045], [0.49, 0.5, 0.51, 0.52],
            [0.71, 1.68, 3.09, 4.23], [1.78, 1.5, 2.12, 2.35]),
        row(3, 3, [0.0095, 0.0125, 0.02, 0], [0.3, 0.4, 0.4, 0],
            [1.29, 3.09, 4.23, 0], [1.65, 2.12, 2.35, 0]),
    ])


def _non_sequential_correlation_tree():
    """Correlation tree for :func:`_non_sequential_id_inventory`, listing
    components in yet another (third) row order: component 10 depends on
    component 25 reaching DS2, forced to at least DS1.
    """
    return pd.DataFrame([
        {"ID": 3, "DEPENDENT ON ITEM": "Independent", "DS0": "Independent",
         "DS1": "Independent", "DS2": "Independent", "DS3": "Independent",
         "DS4": "Independent"},
        {"ID": 10, "DEPENDENT ON ITEM": "25", "DS0": "Independent",
         "DS1": "Independent", "DS2": "DS1", "DS3": "DS1", "DS4": "DS1"},
        {"ID": 25, "DEPENDENT ON ITEM": "Independent", "DS0": "Independent",
         "DS1": "Independent", "DS2": "Independent", "DS3": "Independent",
         "DS4": "Independent"},
    ])


def _two_group_inventory():
    """A 2-component, 2-performance-group synthetic inventory (1 PSD/S row
    + 1 PFA/NS row), used only to verify that generate() slices
    'damage_states' (and other per-group cache entries) down to just the
    components in that group. Not representative of real usage, where PSD
    and PFA data are always kept in separate slfgenerator instances.
    """
    return pd.DataFrame([
        {"Component ID": 1, "Description": "Wall", "EDP": "PSD",
         "Typology": "S", "Performance Group": 1, "Quantity": 1,
         "Damage States": 1, "DS1, Median": 0.01,
         "DS1, Total Dispersion": 0.3, "DS1, Cost": 100.0,
         "DS1, Cost Dispersion": 0.3, "DS1, Best Fit": "Normal"},
        {"Component ID": 2, "Description": "Ceiling", "EDP": "PFA",
         "Typology": "NS", "Performance Group": 2, "Quantity": 1,
         "Damage States": 1, "DS1, Median": 1.0,
         "DS1, Total Dispersion": 0.4, "DS1, Cost": 200.0,
         "DS1, Cost Dispersion": 0.3, "DS1, Best Fit": "Normal"},
    ])


# ---------------------------------------------------------------------------
# Initialisation and validation tests
# ---------------------------------------------------------------------------

class TestSLFGeneratorInit(unittest.TestCase):

    def test_valid_psd_inputs_accepted(self):
        model = _make_model(edp="PSD")
        self.assertEqual(model.edp, "psd")
        self.assertTrue(model.grouping_flag)

    def test_valid_pfa_inputs_accepted(self):
        model = _make_model(edp="PFA")
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

    def test_component_groups_populated(self):
        model = _make_model()
        self.assertGreater(len(model.component_groups), 0)

    def test_regression_methods_and_attribute_removed(self):
        model = _make_model()
        self.assertFalse(hasattr(model, "regression"))
        self.assertFalse(hasattr(model, "perform_regression"))
        self.assertFalse(hasattr(model, "_fit_regression"))
        self.assertFalse(hasattr(model, "estimate_accuracy"))

    def test_id_to_pos_matches_row_order(self):
        model = _make_model()
        self.assertEqual(model.id_to_pos, {1: 0, 2: 1, 3: 2})

    def test_duplicate_component_id_after_autofill_raises(self):
        """An explicit ID that collides with an auto-assigned one (for a
        missing ID elsewhere in the same inventory) must be rejected --
        this is a real gap left by the pre-fill-only duplicate check in
        _validate_component_data_schema.
        """
        cd = os.path.dirname(__file__)
        df = pd.read_csv(os.path.join(cd, "test_data", "slf_inventory.csv"))
        df["Component ID"] = [2, np.nan, 3]  # row 1 auto-fills to 2, collides with row 0
        with self.assertRaises(ValueError):
            slfgenerator(component_data=df, edp="PSD", realizations=10)


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

    def test_fragility_ids_keyed_by_component_id(self):
        fragilities, _, _ = self.model.fragility_function()
        self.assertEqual(set(fragilities["IDs"]), {1, 2, 3})

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
            for key in ("edp", "edp_range", "slf_16th", "slf", "slf_84th"):
                self.assertIn(key, group_out)

    def test_cache_has_required_keys(self):
        for group_cache in self.cache.values():
            for key in (
                "component", "fragilities", "total_loss_storey",
                "total_loss_storey_ratio", "repair_cost", "damage_states",
                "edp", "empirical_16th", "empirical_median",
                "empirical_84th",
            ):
                self.assertIn(key, group_cache)

    def test_slf_length_matches_edp_range(self):
        for group_out in self.out.values():
            n = len(group_out["edp_range"])
            for key in ("slf_16th", "slf", "slf_84th"):
                self.assertEqual(len(group_out[key]), n)

    def test_empirical_median_shape(self):
        for group_cache in self.cache.values():
            median = group_cache["empirical_median"]
            self.assertEqual(len(median), len(self.model.edp_range))

    def test_out_and_cache_percentiles_are_consistent(self):
        """out[...]['slf_*'] and cache[...]['empirical_*'] are built from a
        single shared computation in generate() -- they must be bit-for-bit
        identical, not merely close.
        """
        for key in self.out:
            np.testing.assert_array_equal(
                np.asarray(self.out[key]["slf_16th"]),
                self.cache[key]["empirical_16th"],
            )
            np.testing.assert_array_equal(
                np.asarray(self.out[key]["slf"]),
                self.cache[key]["empirical_median"],
            )
            np.testing.assert_array_equal(
                np.asarray(self.out[key]["slf_84th"]),
                self.cache[key]["empirical_84th"],
            )


# ---------------------------------------------------------------------------
# Correlation tree tests
# ---------------------------------------------------------------------------

class TestSLFGeneratorCorrelationTree(unittest.TestCase):

    def _model_with_tree(self, **overrides):
        return _make_model(correlation_tree=_correlation_tree_fixture(),
                           **overrides)

    def test_matrix_built_correctly(self):
        model = self._model_with_tree()
        expected = np.array([
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 1],
            [3, 0, 0, 0, 0, 0],
        ], dtype=float)
        np.testing.assert_array_equal(model.matrix, expected)
        np.testing.assert_array_equal(model.correlation_item_ids, [1, 2, 3])

    def test_validate_ds_dependence_forces_min_ds(self):
        model = self._model_with_tree(realizations=1)
        damage_state = {
            1: {0: np.array([0, 1, 2, 3, 4])},
            2: {0: np.array([0, 0, 0, 0, 0])},
            3: {0: np.array([0, 0, 0, 0, 0])},
        }
        result = model.validate_ds_dependence(damage_state)
        np.testing.assert_array_equal(result[2][0], [0, 0, 1, 1, 1])
        np.testing.assert_array_equal(result[1][0], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(result[3][0], [0, 0, 0, 0, 0])

    def test_generate_with_correlation_tree_runs_end_to_end(self):
        model = self._model_with_tree(realizations=100)
        out, cache = model.generate()
        self.assertGreater(len(out), 0)
        for group_out in out.values():
            n = len(group_out["edp_range"])
            for key in ("slf_16th", "slf", "slf_84th"):
                self.assertEqual(len(group_out[key]), n)

    def test_correlation_tree_missing_id_raises(self):
        cd = os.path.dirname(__file__)
        df = pd.read_csv(os.path.join(cd, "test_data", "slf_inventory.csv"))
        tree = _correlation_tree_fixture()
        tree = tree[tree["ID"] != 2]  # drop component 2's row entirely
        with self.assertRaises(ValueError):
            slfgenerator(component_data=df, edp="PSD",
                        correlation_tree=tree, realizations=10)

    def test_correlation_tree_works_with_non_sequential_ids(self):
        """Proof test for the ID-based rearchitecture: component IDs are
        non-sequential (25, 10, 3) and the inventory and correlation tree
        list them in two different scrambled row orders. The old
        positional code (dependent ID = row position + 1) would have
        silently applied the dependency to the wrong component here.
        """
        df = _non_sequential_id_inventory()
        tree = _non_sequential_correlation_tree()
        model = slfgenerator(component_data=df, edp="PSD",
                             correlation_tree=tree, realizations=1)
        np.testing.assert_array_equal(
            model.correlation_item_ids, [25, 10, 3]
        )

        damage_state = {
            25: {0: np.array([0, 1, 2, 3, 4])},
            10: {0: np.array([0, 0, 0, 0, 0])},
            3:  {0: np.array([0, 0, 0, 0, 0])},
        }
        result = model.validate_ds_dependence(damage_state)
        np.testing.assert_array_equal(result[10][0], [0, 0, 1, 1, 1])
        np.testing.assert_array_equal(result[25][0], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(result[3][0], [0, 0, 0, 0, 0])

        model_full = slfgenerator(component_data=df, edp="PSD",
                                  correlation_tree=tree, realizations=30)
        out, cache = model_full.generate()
        self.assertGreater(len(out), 0)


# ---------------------------------------------------------------------------
# Group-slicing test
# ---------------------------------------------------------------------------

class TestSLFGeneratorGroupSlicing(unittest.TestCase):

    def test_damage_states_cache_is_group_sliced(self):
        model = slfgenerator(component_data=_two_group_inventory(),
                             edp="PSD", realizations=10)
        out, cache = model.generate()
        self.assertEqual(len(cache), 2)
        for group_cache in cache.values():
            expected_ids = set(
                group_cache["component"]["Component ID"].tolist()
            )
            self.assertEqual(set(group_cache["damage_states"]), expected_ids)
            self.assertNotEqual(set(group_cache["damage_states"]), {1, 2})


if __name__ == "__main__":
    unittest.main()
