import os
import unittest
import numpy as np
import pandas as pd
from scipy import stats

from openquake.vmtk.slfgenerator import slfgenerator


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

def _inventory_file():
    cd = os.path.dirname(__file__)
    return pd.read_csv(os.path.join(cd, "test_data", "slf_inventory.csv"))


def _make_model(edp="PSD", **overrides):
    """Construct a :class:`slfgenerator` with sensible test defaults."""
    kwargs = dict(
        component_data=_inventory_file(),
        edp=edp,
        grouping_flag=True,
        conversion=1.0,
        realizations=200,
        seed=1,
    )
    kwargs.update(overrides)
    return slfgenerator(**kwargs)


def _row(cid, medians, betas, costs, cost_disps, fits, quantity=1.0,
         hierarchy=None, probabilities=None, correlated=None, edp="PSD",
         typology="NS", group=1, replacement=None):
    """One inventory row with up to five damage states."""
    n = len(costs)
    r = {"Component ID": cid, "Description": f"C{cid}", "EDP": edp,
         "Typology": typology, "Performance Group": group,
         "Quantity": quantity, "Damage States": n}
    for k in range(5):
        inside = k < n
        r[f"DS{k + 1}, Median"] = medians[k] if inside and k < len(medians) else np.nan
        r[f"DS{k + 1}, Total Dispersion"] = betas[k] if inside and k < len(betas) else np.nan
        r[f"DS{k + 1}, Cost"] = costs[k] if inside else np.nan
        r[f"DS{k + 1}, Cost Dispersion"] = cost_disps[k] if inside else np.nan
        r[f"DS{k + 1}, Best Fit"] = fits[k] if inside else np.nan
        r[f"DS{k + 1}, Probability"] = (probabilities[k] if probabilities
                                        and inside else np.nan)
    if hierarchy is not None:
        r["DS Hierarchy"] = hierarchy
    if correlated is not None:
        r["Correlated"] = correlated
    if replacement is not None:
        r["Replacement Cost"] = replacement
    return r


def _p58_inventory(correlated="NO"):
    """Sequential, mutually exclusive and simultaneous components, with
    several units each; component 2 depends on component 1."""
    return pd.DataFrame([
        _row(1, [0.005, 0.01, 0.02], [0.5, 0.45, 0.4], [10.0, 30.0, 60.0],
             [0.3, 0.4, 0.2], ["Lognormal", "Normal", "Lognormal"],
             quantity=6.0, correlated=correlated),
        _row(2, [0.004, 0.012, 0.012], [0.6, 0.4, 0.4], [5.0, 20.0, 45.0],
             [0.4, 0.3, 0.3], ["Normal", "Lognormal", "Normal"],
             quantity=12.5, hierarchy="Seq(DS1,MutEx(DS2,DS3))",
             probabilities=[1.0, 0.8, 0.2], correlated=correlated),
        _row(3, [0.008] * 4, [0.45] * 4, [8.0, 37.0, 32.0, 5.0],
             [0.8, 0.28, 0.41, 0.49], ["Lognormal", "Normal", "Normal", "Normal"],
             quantity=2.0, hierarchy="Simul(DS1,DS2,DS3,DS4)",
             probabilities=[0.26, 0.79, 0.68, 0.17], correlated=correlated),
    ])


def _p58_tree():
    return pd.DataFrame({
        "ID": [1, 2, 3],
        "DEPENDENT ON ITEM": ["Independent", "1", "Independent"],
        "DS0": ["Independent"] * 3,
        "DS1": ["Independent"] * 3,
        "DS2": ["Independent", "DS1", "Independent"],
        "DS3": ["Independent", "DS2", "Independent"],
    })


def _correlation_tree_fixture():
    """Correlation tree for the 3-component test fixture (slf_inventory.csv).

    Component 2 (EBCJ) depends on component 1 (IBCJ) reaching DS2 -- once
    component 1 reaches DS2, DS3, or DS4, component 2 is forced to at
    least DS1. Component 3 (Column) is independent.
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
    """Non-sequential Component IDs (25, 10, 3) in scrambled row order."""
    return pd.DataFrame([
        _row(25, [0.0074, 0.025, 0.046, 0.056], [0.3, 0.24, 0.25, 0.25],
             [0.71, 1.68, 3.09, 4.23], [1.78, 1.5, 2.12, 2.35], ["Lognormal"] * 4),
        _row(10, [0.0057, 0.021, 0.037, 0.045], [0.49, 0.5, 0.51, 0.52],
             [0.71, 1.68, 3.09, 4.23], [1.78, 1.5, 2.12, 2.35], ["Lognormal"] * 4),
        _row(3, [0.0095, 0.0125, 0.02], [0.3, 0.4, 0.4],
             [1.29, 3.09, 4.23], [1.65, 2.12, 2.35], ["Lognormal"] * 3),
    ])


def _non_sequential_correlation_tree():
    """Component 10 depends on component 25 reaching DS2, forced to DS1;
    rows in a third order."""
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
    """1 PSD/S row + 1 PFA/NS row in two performance groups."""
    return pd.DataFrame([
        _row(1, [0.01], [0.3], [100.0], [0.3], ["Normal"], typology="S",
             group=1),
        _row(2, [1.0], [0.4], [200.0], [0.3], ["Normal"], edp="PFA",
             typology="NS", group=2),
    ])


def _z_scores(out, cache, n):
    """(Monte Carlo mean - exact mean) / standard error, per EDP."""
    z = []
    for key, rec in out.items():
        se = np.asarray(rec["slf_std"]) / np.sqrt(n)
        diff = cache[key]["simulated_mean"] - np.asarray(rec["slf"])
        z.append(np.divide(diff, se, out=np.zeros_like(diff), where=se > 0))
    return np.concatenate(z)


# ---------------------------------------------------------------------------
# Initialisation and validation
# ---------------------------------------------------------------------------

class TestSLFGeneratorInit(unittest.TestCase):

    def test_valid_psd_inputs_accepted(self):
        model = _make_model(edp="PSD")
        self.assertEqual(model.edp, "psd")
        self.assertTrue(model.grouping_flag)

    def test_valid_pfa_inputs_accepted(self):
        self.assertEqual(_make_model(edp="PFA").edp, "pfa")

    def test_invalid_edp_raises(self):
        with self.assertRaises(ValueError):
            _make_model(edp="SAV")

    def test_invalid_settings_raise(self):
        for bad in (dict(replacement_cost=0.0), dict(replacement_cost=-1.0),
                    dict(realizations=1), dict(conversion=0.0)):
            with self.assertRaises(ValueError, msg=bad):
                _make_model(**bad)

    def test_edp_range_default_starts_at_zero(self):
        model = _make_model(edp="PSD")
        self.assertEqual(model.edp_range[0], 0.0)
        self.assertAlmostEqual(model.edp_range[-1], 0.5)
        self.assertEqual(len(model.edp_range), 501)

    def test_custom_edp_range_is_copied_not_modified(self):
        custom = np.linspace(0, 0.1, 50)
        model = _make_model(edp="PSD", edp_range=custom)
        self.assertEqual(len(model.edp_range), 50)
        self.assertEqual(custom[0], 0.0)
        self.assertIsNot(model.edp_range, custom)

    def test_negative_edp_range_raises(self):
        with self.assertRaises(ValueError):
            _make_model(edp_range=[-0.01, 0.01])

    def test_component_data_not_modified(self):
        df = _inventory_file()
        before = df.copy()
        slfgenerator(df, "PSD", realizations=20, seed=0).generate()
        pd.testing.assert_frame_equal(df, before)

    def test_component_groups_populated(self):
        self.assertGreater(len(_make_model().component_groups), 0)

    def test_grouping_flag_false_gives_one_group(self):
        out, _ = _make_model(grouping_flag=False).generate()
        self.assertEqual(list(out), ["psd"])

    def test_id_to_pos_matches_row_order(self):
        self.assertEqual(_make_model().id_to_pos, {1: 0, 2: 1, 3: 2})

    def test_duplicate_component_id_after_autofill_raises(self):
        df = _inventory_file()
        df["Component ID"] = [2, np.nan, 3]  # row 1 auto-fills to 2
        with self.assertRaises(ValueError):
            slfgenerator(component_data=df, edp="PSD", realizations=10)


class TestInventoryParsing(unittest.TestCase):

    def _model(self, rows, **kwargs):
        kwargs.setdefault("realizations", 50)
        kwargs.setdefault("seed", 0)
        return slfgenerator(pd.DataFrame(rows), "PSD", **kwargs)

    def test_default_hierarchy_is_sequential(self):
        model = _make_model()
        ls = model.components[1]["limit_states"]
        self.assertEqual([x["kind"] for x in ls], ["single"] * 4)
        self.assertEqual([x["damage_states"] for x in ls], [[1], [2], [3], [4]])

    def test_mutex_and_simul_hierarchies(self):
        model = slfgenerator(_p58_inventory(), "PSD", realizations=20)
        ls2 = model.components[2]["limit_states"]
        self.assertEqual([x["kind"] for x in ls2], ["single", "mutex"])
        np.testing.assert_allclose(ls2[1]["p_outcome"], [0.8, 0.2])
        ls3 = model.components[3]["limit_states"]
        self.assertEqual(len(ls3), 1)
        self.assertEqual(ls3[0]["kind"], "simul")
        self.assertEqual(len(ls3[0]["outcomes"]), 15)
        self.assertAlmostEqual(ls3[0]["p_outcome"].sum(), 1.0)
        # P(DS1 alone | at least one) as in the FEMA P-58 database
        self.assertAlmostEqual(ls3[0]["p_outcome"][0], 0.015126, places=6)

    def test_bad_hierarchies_raise(self):
        base = dict(medians=[0.01, 0.02], betas=[0.4, 0.4], costs=[1.0, 2.0],
                    cost_disps=[0.1, 0.1], fits=["Normal"] * 2)
        for hierarchy in ("Seq(DS2,DS1)", "Seq(DS1)", "Par(DS1,DS2)",
                          "Seq(DS1,DS2,DS3)"):
            with self.assertRaises(ValueError, msg=hierarchy):
                self._model([_row(1, hierarchy=hierarchy, **base)])

    def test_mutex_probabilities_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            self._model([_row(1, [0.01, 0.01], [0.4, 0.4], [1.0, 2.0],
                              [0.1, 0.1], ["Normal"] * 2,
                              hierarchy="MutEx(DS1,DS2)",
                              probabilities=[0.5, 0.3])])

    def test_simul_probabilities_must_be_in_unit_interval(self):
        with self.assertRaises(ValueError):
            self._model([_row(1, [0.01, 0.01], [0.4, 0.4], [1.0, 2.0],
                              [0.1, 0.1], ["Normal"] * 2,
                              hierarchy="Simul(DS1,DS2)",
                              probabilities=[0.5, 0.0])])

    def test_limit_state_parameters_must_agree(self):
        with self.assertRaises(ValueError):
            self._model([_row(1, [0.01, 0.02], [0.4, 0.4], [1.0, 2.0],
                              [0.1, 0.1], ["Normal"] * 2,
                              hierarchy="MutEx(DS1,DS2)",
                              probabilities=[0.5, 0.5])])

    def test_shared_limit_state_parameters_may_be_blank(self):
        model = self._model([_row(1, [0.01], [0.4], [1.0, 2.0], [0.1, 0.1],
                                  ["Normal"] * 2, hierarchy="MutEx(DS1,DS2)",
                                  probabilities=[0.5, 0.5])])
        self.assertEqual(model.components[1]["n_ls"], 1)

    def test_decreasing_medians_raise(self):
        with self.assertRaises(ValueError):
            self._model([_row(1, [0.02, 0.01], [0.4, 0.4], [1.0, 2.0],
                              [0.1, 0.1], ["Normal"] * 2)])

    def test_invalid_best_fit_and_correlated_raise(self):
        with self.assertRaises(ValueError):
            self._model([_row(1, [0.01], [0.4], [1.0], [0.1], ["Weibull"])])
        with self.assertRaises(ValueError):
            self._model([_row(1, [0.01], [0.4], [1.0], [0.1], ["Normal"],
                              correlated="maybe")])

    def test_blocks_follow_the_correlated_flag(self):
        model = slfgenerator(_p58_inventory("NO"), "PSD", realizations=20)
        self.assertEqual([model.components[i]["blocks"] for i in (1, 2, 3)],
                         [6, 13, 2])
        self.assertAlmostEqual(model.components[2]["block_quantity"], 12.5 / 13)
        model = slfgenerator(_p58_inventory("YES"), "PSD", realizations=20)
        self.assertEqual([model.components[i]["blocks"] for i in (1, 2, 3)],
                         [1, 1, 1])


# ---------------------------------------------------------------------------
# Fragility and exact expectation
# ---------------------------------------------------------------------------

class TestFragilityAndExpectation(unittest.TestCase):

    def test_fragility_keyed_by_component_and_limit_state(self):
        fragilities = _make_model().fragility_function()
        self.assertEqual(set(fragilities["IDs"]), {1, 2, 3})
        self.assertEqual(list(fragilities["IDs"][3]), ["LS1", "LS2", "LS3"])

    def test_fragility_bounded_ordered_and_zero_at_zero(self):
        fragilities = _make_model(edp_range=np.linspace(0, 0.2, 41)).fragility_function()
        for curves in fragilities["IDs"].values():
            stack = np.array(list(curves.values()))
            self.assertTrue(np.all((stack >= 0) & (stack <= 1)))
            self.assertTrue(np.all(stack[:, 0] == 0))
            self.assertTrue(np.all(np.diff(stack, axis=0) <= 0))
            self.assertTrue(np.all(np.diff(stack, axis=1) >= 0))

    def test_lognormal_fragility_uses_the_median(self):
        model = slfgenerator(pd.DataFrame([_row(1, [0.01], [0.4], [1.0], [0.0],
                                                ["Normal"])]),
                             "PSD", edp_range=[0.01, 0.02], realizations=10)
        poe = model.fragility_function()["IDs"][1]["LS1"]
        self.assertAlmostEqual(poe[0], 0.5)
        self.assertAlmostEqual(poe[1], stats.norm.cdf(np.log(2) / 0.4))

    def test_expected_loss_single_component(self):
        edp = np.linspace(0, 0.05, 11)
        inv = pd.DataFrame([_row(1, [0.01], [0.4], [100.0], [0.5], ["Lognormal"],
                                 quantity=3.0, replacement=500.0)])
        out, _ = slfgenerator(inv, "PSD", edp_range=edp, realizations=50,
                              seed=0).generate()
        poe = np.zeros_like(edp)
        poe[1:] = stats.norm.cdf(np.log(edp[1:] / 0.01) / 0.4)
        expected = 3.0 * 100.0 * np.exp(0.5 * 0.5 ** 2) * poe / (3.0 * 500.0)
        np.testing.assert_allclose(out["PSD, NS"]["slf"], expected, rtol=1e-12)
        self.assertEqual(out["PSD, NS"]["replacement_cost"], 1500.0)

    def test_truncated_normal_cost_mean(self):
        inv = pd.DataFrame([_row(1, [0.001], [0.1], [100.0], [0.8], ["Normal"])])
        model = slfgenerator(inv, "PSD", edp_range=[1.0], realizations=10)
        truncated = stats.truncnorm(-1 / 0.8, np.inf, loc=100.0, scale=80.0)
        self.assertAlmostEqual(model.components[1]["mean_cost"][0],
                               truncated.mean(), places=8)

    def test_mutex_and_simul_expected_costs(self):
        model = slfgenerator(_p58_inventory(), "PSD", realizations=20)
        c2 = model.components[2]
        self.assertAlmostEqual(c2["limit_states"][1]["mean_cost"],
                               0.8 * c2["mean_cost"][1] + 0.2 * c2["mean_cost"][2])
        c3 = model.components[3]
        p = np.array([0.26, 0.79, 0.68, 0.17])
        self.assertAlmostEqual(c3["limit_states"][0]["mean_cost"],
                               p @ c3["mean_cost"] / (1 - np.prod(1 - p)))

    def test_default_replacement_cost_is_the_largest_outcome(self):
        model = slfgenerator(_p58_inventory(), "PSD", realizations=20)
        c3 = model.components[3]
        self.assertAlmostEqual(c3["replacement"], c3["mean_cost"].sum())
        c2 = model.components[2]
        self.assertAlmostEqual(c2["replacement"], c2["mean_cost"].max())

    def test_replacement_cost_argument_and_conversion(self):
        inv = _p58_inventory()
        out1, _ = slfgenerator(inv, "PSD", realizations=50, seed=2).generate()
        out2, _ = slfgenerator(inv, "PSD", realizations=50, seed=2,
                               conversion=2.0).generate()
        np.testing.assert_allclose(out1["PSD, NS"]["slf"], out2["PSD, NS"]["slf"])
        self.assertAlmostEqual(out2["PSD, NS"]["replacement_cost"],
                               2 * out1["PSD, NS"]["replacement_cost"])
        out3, _ = slfgenerator(inv, "PSD", realizations=50, seed=2,
                               replacement_cost=1.0).generate()
        np.testing.assert_allclose(
            out3["PSD, NS"]["slf"],
            np.asarray(out1["PSD, NS"]["slf"]) * out1["PSD, NS"]["replacement_cost"])

    def test_limit_state_probabilities_sum_to_one(self):
        model = slfgenerator(_p58_inventory(), "PSD", correlation_tree=_p58_tree(),
                             realizations=20)
        for p in model.limit_state_probabilities().values():
            np.testing.assert_allclose(p.sum(axis=0), 1.0)
            self.assertTrue(np.all(p >= -1e-12))


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

class TestMonteCarlo(unittest.TestCase):

    def test_simulated_mean_matches_exact_mean(self):
        n = 4000
        for tree in (None, _p58_tree()):
            for flag in ("YES", "NO"):
                out, cache = slfgenerator(
                    _p58_inventory(flag), "PSD", correlation_tree=tree,
                    edp_range=np.linspace(0, 0.05, 26), realizations=n,
                    seed=11).generate()
                z = _z_scores(out, cache, n)
                self.assertLess(np.abs(z).max(), 4.5, (flag, tree is not None))

    def test_reproducible_with_seed(self):
        a, _ = _make_model(seed=5).generate()
        b, _ = _make_model(seed=5).generate()
        c, _ = _make_model(seed=6).generate()
        self.assertEqual(a, b)
        self.assertNotEqual(a["PSD, S"]["slf_std"], c["PSD, S"]["slf_std"])
        self.assertEqual(a["PSD, S"]["slf"], c["PSD, S"]["slf"])   # exact mean

    def test_realisation_losses_increase_with_edp(self):
        """Capacities are held over the EDP range, so with costs that grow
        with the damage state every realisation is non-decreasing."""
        inv = pd.DataFrame([_row(1, [0.005, 0.01], [0.5, 0.4], [10.0, 30.0],
                                 [0.0, 0.0], ["Lognormal"] * 2, quantity=8,
                                 correlated="NO")])
        _, cache = slfgenerator(inv, "PSD", realizations=300, seed=3,
                                edp_range=np.linspace(0, 0.05, 51)).generate()
        self.assertTrue(np.all(np.diff(cache["PSD, NS"]["total_loss_storey"],
                                       axis=1) >= 0))

    def test_independent_units_reduce_dispersion_not_mean(self):
        rows = lambda flag: pd.DataFrame([_row(1, [0.01], [0.5], [100.0], [0.0],
                                               ["Normal"], quantity=50,
                                               correlated=flag)])
        edp = [0.005, 0.01, 0.02]
        out_yes, _ = slfgenerator(rows("YES"), "PSD", edp_range=edp,
                                  realizations=4000, seed=4).generate()
        out_no, _ = slfgenerator(rows("NO"), "PSD", edp_range=edp,
                                 realizations=4000, seed=4).generate()
        np.testing.assert_allclose(out_yes["PSD, NS"]["slf"], out_no["PSD, NS"]["slf"])
        p = stats.norm.cdf(np.log(np.array(edp) / 0.01) / 0.5)
        # one unit cost, no cost dispersion: binomial number of damaged units
        np.testing.assert_allclose(out_no["PSD, NS"]["slf_std"],
                                   np.sqrt(p * (1 - p) / 50), rtol=0.08)
        np.testing.assert_allclose(out_yes["PSD, NS"]["slf_std"],
                                   np.sqrt(p * (1 - p)), rtol=0.08)

    def test_mutually_exclusive_outcome_frequencies(self):
        inv = pd.DataFrame([_row(1, [1e-4], [0.1], [1.0, 2.0, 4.0], [0, 0, 0],
                                 ["Normal"] * 3, hierarchy="MutEx(DS1,DS2,DS3)",
                                 probabilities=[0.5, 0.3, 0.2],
                                 replacement=1.0)])
        _, cache = slfgenerator(inv, "PSD", edp_range=[1.0], realizations=20000,
                                seed=8).generate()
        values = cache["PSD, NS"]["total_loss_storey"][:, 0]
        freq = [np.mean(values == v) for v in (1.0, 2.0, 4.0)]
        np.testing.assert_allclose(freq, [0.5, 0.3, 0.2], atol=0.015)

    def test_simultaneous_outcome_frequencies(self):
        p = np.array([0.26, 0.79, 0.68, 0.17])
        inv = pd.DataFrame([_row(1, [1e-4], [0.1], [1.0, 2.0, 4.0, 8.0],
                                 [0] * 4, ["Normal"] * 4,
                                 hierarchy="Simul(DS1,DS2,DS3,DS4)",
                                 probabilities=list(p), replacement=1.0)])
        _, cache = slfgenerator(inv, "PSD", edp_range=[1.0], realizations=40000,
                                seed=9).generate()
        mask = cache["PSD, NS"]["total_loss_storey"][:, 0].astype(int)
        self.assertTrue(np.all((mask >= 1) & (mask <= 15)))   # never none
        bits = (mask[:, None] >> np.arange(4)) & 1
        np.testing.assert_allclose(bits.mean(axis=0), p / (1 - np.prod(1 - p)),
                                   atol=0.01)
        # the damage states occur independently of one another
        joint = np.mean(bits[:, 1] & bits[:, 2])
        self.assertAlmostEqual(joint, p[1] * p[2] / (1 - np.prod(1 - p)), delta=0.01)


# ---------------------------------------------------------------------------
# generate() outputs
# ---------------------------------------------------------------------------

class TestSLFGeneratorGenerate(unittest.TestCase):

    def setUp(self):
        self.model = _make_model()
        self.out, self.cache = self.model.generate()

    def test_out_keys_match_cache_keys(self):
        self.assertEqual(set(self.out), set(self.cache))

    def test_out_has_required_keys(self):
        for rec in self.out.values():
            for key in ("edp", "edp_range", "slf", "slf_std", "slf_cov",
                        "slf_16th", "slf_50th", "slf_84th", "replacement_cost"):
                self.assertIn(key, rec)

    def test_cache_has_required_keys(self):
        for group_cache in self.cache.values():
            for key in ("component", "fragilities", "limit_state_probabilities",
                        "expected_component_loss", "total_loss_storey",
                        "total_loss_storey_ratio", "simulated_mean",
                        "replacement_cost", "edp", "empirical_16th",
                        "empirical_median", "empirical_84th"):
                self.assertIn(key, group_cache)

    def test_lengths_and_shapes(self):
        n = len(self.model.edp_range)
        for key, rec in self.out.items():
            for field in ("slf", "slf_std", "slf_cov", "slf_16th", "slf_50th",
                          "slf_84th"):
                self.assertEqual(len(rec[field]), n)
            self.assertEqual(self.cache[key]["total_loss_storey_ratio"].shape,
                             (self.model.realizations, n))

    def test_out_and_cache_percentiles_are_identical(self):
        for key in self.out:
            np.testing.assert_array_equal(self.out[key]["slf_16th"],
                                          self.cache[key]["empirical_16th"])
            np.testing.assert_array_equal(self.out[key]["slf_50th"],
                                          self.cache[key]["empirical_median"])
            np.testing.assert_array_equal(self.out[key]["slf_84th"],
                                          self.cache[key]["empirical_84th"])

    def test_cov_is_std_over_mean(self):
        for rec in self.out.values():
            mean, std = np.asarray(rec["slf"]), np.asarray(rec["slf_std"])
            positive = mean > 0
            np.testing.assert_allclose(np.asarray(rec["slf_cov"])[positive],
                                       std[positive] / mean[positive])
            self.assertTrue(np.all(np.asarray(rec["slf_cov"])[~positive] == 0))

    def test_mean_is_sum_of_component_expectations(self):
        for key, rec in self.out.items():
            total = sum(self.cache[key]["expected_component_loss"].values())
            np.testing.assert_allclose(
                rec["slf"], total / self.cache[key]["replacement_cost"])


# ---------------------------------------------------------------------------
# Correlation tree
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

    def test_apply_correlation_tree_forces_min_state(self):
        model = self._model_with_tree()
        states = {1: np.array([[0, 1, 2, 3, 4]]), 2: np.zeros((1, 5), int),
                  3: np.zeros((1, 5), int)}
        result = model.apply_correlation_tree(states)
        np.testing.assert_array_equal(result[2][0], [0, 0, 1, 1, 1])
        np.testing.assert_array_equal(result[1][0], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(result[3][0], [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(states[2][0], [0, 0, 0, 0, 0])

    def test_no_tree_returns_states_unchanged(self):
        states = {1: np.zeros((1, 3), int)}
        self.assertIs(_make_model().apply_correlation_tree(states), states)

    def test_units_are_hosted_in_proportion(self):
        model = slfgenerator(_p58_inventory("NO"), "PSD",
                             correlation_tree=_p58_tree(), realizations=10)
        # 13 dependent blocks over 6 causation blocks
        hosts = [model.host_block(b, 13, 6) for b in range(13)]
        self.assertEqual(hosts, [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        states = {1: np.array([[3], [0], [0], [0], [0], [1]]),
                  2: np.zeros((13, 1), int), 3: np.zeros((2, 1), int)}
        final = model.apply_correlation_tree(states)[2][:, 0]
        np.testing.assert_array_equal(final, [2, 2, 2] + [0] * 8 + [0, 0])

    def test_tree_raises_the_expected_loss(self):
        edp = np.linspace(0, 0.03, 16)
        base, _ = slfgenerator(_p58_inventory(), "PSD", edp_range=edp,
                               realizations=20).generate()
        tied, _ = slfgenerator(_p58_inventory(), "PSD", edp_range=edp,
                               correlation_tree=_p58_tree(),
                               realizations=20).generate()
        diff = np.asarray(tied["PSD, NS"]["slf"]) - np.asarray(base["PSD, NS"]["slf"])
        self.assertTrue(np.all(diff >= -1e-15))
        self.assertGreater(diff.max(), 0.0)

    def test_chained_dependencies(self):
        tree = _p58_tree()
        tree.loc[2, ["DEPENDENT ON ITEM", "DS2"]] = ["2", "DS1"]
        model = slfgenerator(_p58_inventory(), "PSD", correlation_tree=tree,
                             edp_range=np.linspace(0, 0.03, 16),
                             realizations=4000, seed=12)
        self.assertEqual(model.order, [1, 2, 3])
        out, cache = model.generate()
        self.assertLess(np.abs(_z_scores(out, cache, 4000)).max(), 4.5)

    def test_invalid_trees_raise(self):
        def tree_with(**cells):
            tree = _p58_tree()
            for column, value in cells.items():
                tree.loc[1, column] = value
            return tree

        bad_trees = [
            tree_with(**{"DEPENDENT ON ITEM": "9"}),          # unknown cause
            tree_with(**{"DEPENDENT ON ITEM": "2"}),          # itself
            tree_with(DS0="DS1"),                             # undamaged cause
            tree_with(DS3="DS3"),                             # beyond its LS
            tree_with(DS2="DS2", DS3="DS1"),                  # decreasing
            tree_with(DS2="broken"),
        ]
        cycle = _p58_tree()
        cycle.loc[0, ["DEPENDENT ON ITEM", "DS2"]] = ["2", "DS1"]
        bad_trees.append(cycle)
        forcing_independent = _p58_tree()
        forcing_independent.loc[2, "DS2"] = "DS1"
        bad_trees.append(forcing_independent)
        for tree in bad_trees:
            with self.assertRaises(ValueError):
                slfgenerator(_p58_inventory(), "PSD", correlation_tree=tree,
                             realizations=10)

    def test_columns_beyond_the_causation_states_are_ignored(self):
        tree = _p58_tree()
        tree["DS4"] = ["Independent", "DS2", "Independent"]
        model = slfgenerator(_p58_inventory(), "PSD", correlation_tree=tree,
                             realizations=10)
        np.testing.assert_array_equal(model.causation[2][1], [0, 0, 1, 2])

    def test_correlation_tree_missing_id_raises(self):
        tree = _correlation_tree_fixture()
        tree = tree[tree["ID"] != 2]
        with self.assertRaises(ValueError):
            slfgenerator(component_data=_inventory_file(), edp="PSD",
                         correlation_tree=tree, realizations=10)

    def test_correlation_tree_works_with_non_sequential_ids(self):
        df = _non_sequential_id_inventory()
        tree = _non_sequential_correlation_tree()
        model = slfgenerator(component_data=df, edp="PSD",
                             correlation_tree=tree, realizations=30, seed=0)
        np.testing.assert_array_equal(model.correlation_item_ids, [25, 10, 3])
        states = {25: np.array([[0, 1, 2, 3, 4]]), 10: np.zeros((1, 5), int),
                  3: np.zeros((1, 5), int)}
        result = model.apply_correlation_tree(states)
        np.testing.assert_array_equal(result[10][0], [0, 0, 1, 1, 1])
        np.testing.assert_array_equal(result[3][0], [0, 0, 0, 0, 0])
        out, _ = model.generate()
        self.assertGreater(len(out), 0)


# ---------------------------------------------------------------------------
# Group slicing
# ---------------------------------------------------------------------------

class TestSLFGeneratorGroupSlicing(unittest.TestCase):

    def test_cache_is_group_sliced(self):
        model = slfgenerator(component_data=_two_group_inventory(),
                             edp="PSD", realizations=10, seed=0)
        out, cache = model.generate()
        self.assertEqual(len(cache), 2)
        for group_cache in cache.values():
            expected_ids = set(group_cache["component"]["Component ID"].tolist())
            self.assertEqual(set(group_cache["limit_state_probabilities"]),
                             expected_ids)
            self.assertEqual(set(group_cache["expected_component_loss"]),
                             expected_ids)
            self.assertEqual(len(expected_ids), 1)


if __name__ == "__main__":
    unittest.main()
