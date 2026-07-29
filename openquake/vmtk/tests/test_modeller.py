import os
import shutil
import unittest
import numpy as np
import pytest

pytest.importorskip("openseespy.opensees",
                    reason="openseespy not installed — modeller tests skipped")

from openquake.vmtk.modeller import modeller


# ---------------------------------------------------------------------------
# Shared valid inputs
# ---------------------------------------------------------------------------
NUMBER_STOREYS = 2
STOREY_HEIGHTS = [2.80, 2.80]
FLOOR_MASSES = [0.5979496788391293, 0.448462259129347]
STOREY_DRIFTS = np.array([
    [0.00052664, 0.00421318, 0.02096557, 0.03771796],
    [0.00028185, 0.00225482, 0.01122043, 0.02018604],
])
STOREY_FORCES = np.array([
    [0.11496146, 0.22992293, 0.13795376, 0.1393333],
    [0.06152551, 0.12305102, 0.07383061, 0.07456892],
]) * 9.81
DEGRADATION = True


def make_modeller(**overrides):
    """Return a modeller instance with optional keyword overrides."""
    kwargs = dict(
        number_storeys=NUMBER_STOREYS,
        storey_heights=STOREY_HEIGHTS,
        floor_masses=FLOOR_MASSES,
        storey_drifts=STOREY_DRIFTS,
        storey_forces=STOREY_FORCES,
        degradation=DEGRADATION,
    )
    kwargs.update(overrides)
    return modeller(**kwargs)


# ---------------------------------------------------------------------------
# __init__ validation tests
# ---------------------------------------------------------------------------
class TestModellerInit(unittest.TestCase):

    # --- happy path ---------------------------------------------------------

    def test_valid_inputs_accepted(self):
        m = make_modeller()
        self.assertEqual(m.number_storeys, NUMBER_STOREYS)
        np.testing.assert_array_equal(m.storey_heights, STOREY_HEIGHTS)
        np.testing.assert_array_equal(m.floor_masses, FLOOR_MASSES)
        np.testing.assert_array_equal(m.storey_drifts, STOREY_DRIFTS)
        np.testing.assert_array_equal(m.storey_forces, STOREY_FORCES)
        self.assertEqual(m.degradation, DEGRADATION)

    def test_bilinear_cap_points_accepted(self):
        make_modeller(
            storey_drifts=np.array([[0.001, 0.01], [0.001, 0.01]]),
            storey_forces=np.array([[10.0, 5.0],   [10.0, 5.0]]),
        )

    def test_trilinear_cap_points_accepted(self):
        make_modeller(
            storey_drifts=np.array(
                [[0.001, 0.005, 0.02], [0.001, 0.005, 0.02]]),
            storey_forces=np.array([[10.0, 15.0, 8.0], [10.0, 15.0, 8.0]]),
        )

    def test_single_storey_accepted(self):
        make_modeller(
            number_storeys=1,
            storey_heights=[3.0],
            floor_masses=[1.0],
            storey_drifts=np.array([[0.001, 0.005, 0.02, 0.04]]),
            storey_forces=np.array([[10.0, 20.0, 12.0, 13.0]]),
        )

    def test_degradation_false_accepted(self):
        make_modeller(degradation=False)

    # --- number_storeys -----------------------------------------------------

    def test_number_storeys_zero_raises(self):
        with self.assertRaises(ValueError):
            make_modeller(number_storeys=0)

    def test_number_storeys_negative_raises(self):
        with self.assertRaises(ValueError):
            make_modeller(number_storeys=-1)

    def test_number_storeys_float_raises(self):
        with self.assertRaises(ValueError):
            make_modeller(number_storeys=2.0)

    def test_number_storeys_string_raises(self):
        with self.assertRaises(ValueError):
            make_modeller(number_storeys="2")

    # --- storey_heights -----------------------------------------------------

    def test_storey_heights_scalar_raises(self):
        with self.assertRaises(TypeError):
            make_modeller(storey_heights=2.8)

    def test_storey_heights_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            make_modeller(storey_heights=[2.8])

    def test_storey_heights_zero_value_raises(self):
        with self.assertRaises(ValueError):
            make_modeller(storey_heights=[0.0, 2.8])

    def test_storey_heights_negative_value_raises(self):
        with self.assertRaises(ValueError):
            make_modeller(storey_heights=[-1.0, 2.8])

    # --- floor_masses -------------------------------------------------------

    def test_floor_masses_scalar_raises(self):
        with self.assertRaises(TypeError):
            make_modeller(floor_masses=1.0)

    def test_floor_masses_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            make_modeller(floor_masses=[1.0])

    def test_floor_masses_zero_value_raises(self):
        with self.assertRaises(ValueError):
            make_modeller(floor_masses=[0.0, 1.0])

    def test_floor_masses_negative_value_raises(self):
        with self.assertRaises(ValueError):
            make_modeller(floor_masses=[-0.5, 1.0])

    # --- storey_drifts ------------------------------------------------------

    def test_storey_drifts_wrong_row_count_raises(self):
        with self.assertRaises(ValueError):
            make_modeller(
                storey_drifts=np.array([[0.001, 0.005, 0.02, 0.04]]),
                storey_forces=np.array([[10.0, 20.0, 12.0, 13.0]]),
            )

    def test_storey_drifts_too_many_cap_points_raises(self):
        with self.assertRaises(ValueError):
            make_modeller(
                storey_drifts=np.array([[0.001, 0.005, 0.01, 0.02, 0.04],
                                        [0.001, 0.005, 0.01, 0.02, 0.04]]),
                storey_forces=np.array([[10.0, 15.0, 20.0, 12.0, 13.0],
                                        [10.0, 15.0, 20.0, 12.0, 13.0]]),
            )

    def test_storey_drifts_one_cap_point_raises(self):
        with self.assertRaises(ValueError):
            make_modeller(
                storey_drifts=np.array([[0.001], [0.001]]),
                storey_forces=np.array([[10.0],  [10.0]]),
            )

    def test_storey_drifts_non_positive_raises(self):
        bad = STOREY_DRIFTS.copy()
        bad[0, 0] = 0.0
        with self.assertRaises(ValueError):
            make_modeller(storey_drifts=bad)

    def test_storey_drifts_not_strictly_increasing_raises(self):
        bad = STOREY_DRIFTS.copy()
        bad[0, 2] = bad[0, 1]  # duplicate → not strictly increasing
        with self.assertRaises(ValueError):
            make_modeller(storey_drifts=bad)

    def test_storey_drifts_decreasing_raises(self):
        bad = STOREY_DRIFTS.copy()
        bad[1, 3] = bad[1, 2] - 0.0001  # decreasing in second storey
        with self.assertRaises(ValueError):
            make_modeller(storey_drifts=bad)

    # --- storey_forces ------------------------------------------------------

    def test_storey_forces_wrong_row_count_raises(self):
        with self.assertRaises(ValueError):
            make_modeller(
                storey_forces=np.array([[10.0, 20.0, 12.0, 13.0]]),
            )

    def test_storey_forces_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            make_modeller(
                storey_forces=np.array([[10.0, 20.0, 12.0],
                                        [10.0, 20.0, 12.0]]),
            )

    def test_storey_forces_non_positive_raises(self):
        bad = STOREY_FORCES.copy()
        bad[1, 1] = 0.0
        with self.assertRaises(ValueError):
            make_modeller(storey_forces=bad)

    def test_storey_forces_softening_allowed(self):
        """Forces may decrease after peak (post-peak softening is valid)."""
        make_modeller(
            storey_drifts=np.array([[0.001, 0.005, 0.02, 0.04],
                                    [0.001, 0.005, 0.02, 0.04]]),
            storey_forces=np.array([[10.0, 20.0, 15.0, 5.0],
                                    [10.0, 20.0, 15.0, 5.0]]),
        )

    # --- degradation --------------------------------------------------------

    def test_degradation_int_raises(self):
        with self.assertRaises(TypeError):
            make_modeller(degradation=1)

    def test_degradation_string_raises(self):
        with self.assertRaises(TypeError):
            make_modeller(degradation="True")


# ---------------------------------------------------------------------------
# Analysis method tests
# ---------------------------------------------------------------------------
class TestModellerMethods(unittest.TestCase):

    PHI = np.array([0.65138782, 1.0])
    T_EXPECTED = 0.154

    def setUp(self):
        self.model = make_modeller()
        self.model.compile_model()

        cd = os.path.dirname(__file__)
        acc = np.loadtxt(os.path.join(cd, "test_data", "acceleration.txt"))
        self.fnames = [os.path.join(cd, "test_data", "acceleration.txt")]
        self.dt_gm = 0.005
        self.t_max = 30.0
        self.time_vector = np.arange(len(acc)) * self.dt_gm
        self.temp_folder = os.path.join(cd, "test_data", "temp")
        os.makedirs(self.temp_folder, exist_ok=True)

    def tearDown(self):
        if os.path.isdir(self.temp_folder):
            shutil.rmtree(self.temp_folder)

    def test_compile_model(self):
        self.model.compile_model()

    def test_gravity_analysis(self):
        self.model.do_gravity_analysis()

    def test_modal_analysis_period(self):
        T, phi = self.model.do_modal_analysis(num_modes=3)
        self.assertAlmostEqual(T[0], self.T_EXPECTED, places=4)

    def test_modal_analysis_mode_shape(self):
        T, phi = self.model.do_modal_analysis(num_modes=3)
        # phi[0] is (n_nodes x 3); skip base node (row 0), take x-direction (col 0)
        raw = phi[0][1:, 0]
        # Normalise by the maximum absolute component — OpenSees does not
        # guarantee a consistent eigenvector scale across platforms/versions.
        normalised = raw / np.abs(raw).max()
        np.testing.assert_array_almost_equal(normalised, self.PHI, decimal=4)

    def test_spo_analysis(self):
        self.model.do_spo_analysis(0.01, 5, 1, self.PHI, pFlag=False)

    def test_cpo_analysis(self):
        self.model.do_cpo_analysis(
            0.01, [1, 5, 10], 1, 2, self.PHI, pFlag=False)

    def test_nrha_analysis_returns_13_values(self):
        self.model.do_modal_analysis(num_modes=3)
        result = self.model.do_nrha_analysis(
            self.fnames,
            self.dt_gm,
            sf=9.81,
            t_max=self.t_max,
            dt_ansys=0.001,
            save_animation_path=self.temp_folder,
            pFlag=False,
            xi=0.05,
        )
        self.assertEqual(len(result), 13)

    def test_nrha_analysis_conv_index(self):
        self.model.do_modal_analysis(num_modes=3)
        result = self.model.do_nrha_analysis(
            self.fnames,
            self.dt_gm,
            sf=9.81,
            t_max=self.t_max,
            dt_ansys=0.001,
            pFlag=False,
            xi=0.05,
        )
        _, conv_index, *_ = result
        self.assertIn(conv_index, (0, 1))

    def test_nrha_analysis_peak_drift_shape(self):
        self.model.do_modal_analysis(num_modes=3)
        result = self.model.do_nrha_analysis(
            self.fnames,
            self.dt_gm,
            sf=9.81,
            t_max=self.t_max,
            dt_ansys=0.001,
            pFlag=False,
            xi=0.05,
        )
        _, _, peak_drift, *_ = result
        # peak_drift shape: (n_storeys, n_directions)
        self.assertEqual(peak_drift.shape[0], NUMBER_STOREYS)

    def test_incremental_dynamic_analysis_returns(self):
        self.model.do_modal_analysis(num_modes=3)
        ida_data, ordered_sfs = self.model.do_ida_analysis(
            self.fnames,
            self.dt_gm,
            t_max=self.t_max,
            dt_ansys=0.001,
            initial_sf=0.1,
            hunt_step=2.0,
            max_runs=2,
            pFlag=False,
        )
        self.assertIsInstance(ida_data, dict)
        self.assertIsInstance(ordered_sfs, list)
        self.assertGreater(len(ordered_sfs), 0)

    def test_incremental_dynamic_analysis_data_keys(self):
        self.model.do_modal_analysis(num_modes=3)
        ida_data, ordered_sfs = self.model.do_ida_analysis(
            self.fnames,
            self.dt_gm,
            t_max=self.t_max,
            dt_ansys=0.001,
            initial_sf=0.1,
            hunt_step=2.0,
            max_runs=2,
            pFlag=False,
        )
        # Each scale factor key should map to a result dict
        for sf in ordered_sfs:
            self.assertIn(sf, ida_data)
            self.assertIsInstance(ida_data[sf], dict)

    def test_nrha_analysis_sequences_returns_19_values(self):
        self.model.do_modal_analysis(num_modes=3)
        result = self.model.do_nrha_analysis_sequences(
            self.fnames,
            self.time_vector,
            sf=9.81,
            pFlag=False,
            xi=0.05,
        )
        self.assertEqual(len(result), 19)

    def test_nrha_analysis_sequences_n_sequences(self):
        self.model.do_modal_analysis(num_modes=3)
        result = self.model.do_nrha_analysis_sequences(
            self.fnames,
            self.time_vector,
            sf=9.81,
            pFlag=False,
            xi=0.05,
        )
        n_sequences = result[17]
        boundaries = result[18]
        self.assertIsInstance(n_sequences, int)
        self.assertGreater(n_sequences, 0)
        self.assertEqual(len(boundaries), n_sequences)


if __name__ == "__main__":
    unittest.main()
