import unittest
import numpy as np
import pytest

pytest.importorskip("openseespy.opensees",
                    reason="openseespy not installed — calibration tests skipped")

from openquake.vmtk.calibration import calibration


SDOF_CAPACITY = np.array(
    [[0.00060789, 0.00486316, 0.02420000, 0.04353684],
     [0.10315200, 0.20630401, 0.12378241, 0.12502023]]
).T

NUMBER_STOREYS = 2
STOREY_HEIGHTS = [2.80, 2.80]
T_TARGET = 0.154


class TestCalibration(unittest.TestCase):

    def _make_cal(self, **kwargs):
        defaults = dict(
            nst=NUMBER_STOREYS,
            sdof_capacity=SDOF_CAPACITY,
            is_sos=False,
        )
        defaults.update(kwargs)
        return calibration(**defaults)

    # --- happy path ---------------------------------------------------------

    def test_returns_five_values(self):
        result = self._make_cal().calibrate_model()
        self.assertEqual(len(result), 5)

    def test_floor_masses_sum_to_one(self):
        floor_masses, *_ = self._make_cal().calibrate_model()
        self.assertAlmostEqual(sum(floor_masses), 1.0, places=10)

    def test_floor_masses_length(self):
        floor_masses, *_ = self._make_cal().calibrate_model()
        self.assertEqual(len(floor_masses), NUMBER_STOREYS)

    def test_phi_roof_normalised(self):
        _, _, _, phi, _ = self._make_cal().calibrate_model()
        self.assertAlmostEqual(phi[-1], 1.0, places=10)

    def test_phi_length(self):
        _, _, _, phi, _ = self._make_cal().calibrate_model()
        self.assertEqual(len(phi), NUMBER_STOREYS)

    def test_output_shapes(self):
        _, storey_drifts, storey_forces, _, _ = (
            self._make_cal().calibrate_model()
        )
        n_pts = SDOF_CAPACITY.shape[0]
        self.assertEqual(storey_drifts.shape, (NUMBER_STOREYS, n_pts))
        self.assertEqual(storey_forces.shape, (NUMBER_STOREYS, n_pts))

    def test_t_target_matches(self):
        _, _, _, _, metadata = self._make_cal().calibrate_model()
        self.assertAlmostEqual(metadata["T_target"], T_TARGET, places=3)

    def test_storey_drifts_positive_and_increasing(self):
        _, storey_drifts, _, _, _ = self._make_cal().calibrate_model()
        for row in storey_drifts:
            self.assertTrue(np.all(row > 0))
            self.assertTrue(np.all(np.diff(row) > 0))

    def test_storey_forces_positive(self):
        _, _, storey_forces, _, _ = self._make_cal().calibrate_model()
        self.assertTrue(np.all(storey_forces > 0))

    def test_with_storey_heights_triggers_spo(self):
        """Providing storey_heights triggers OpenSees SPO verification."""
        result = self._make_cal(
            storey_heights=STOREY_HEIGHTS
        ).calibrate_model()
        self.assertEqual(len(result), 5)
        _, _, _, _, metadata = result
        self.assertIn("T_achieved", metadata)

    def test_soft_storey_flag(self):
        result = self._make_cal(is_sos=True).calibrate_model()
        self.assertEqual(len(result), 5)

    # --- __init__ validation ------------------------------------------------

    def test_invalid_nst_raises(self):
        with self.assertRaises(ValueError):
            calibration(nst=0, sdof_capacity=SDOF_CAPACITY)

    def test_nst_float_raises(self):
        with self.assertRaises(ValueError):
            calibration(nst=2.0, sdof_capacity=SDOF_CAPACITY)

    def test_sdof_capacity_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            calibration(nst=NUMBER_STOREYS,
                        sdof_capacity=np.array([0.001, 0.1]))

    def test_sdof_capacity_too_few_points_raises(self):
        with self.assertRaises(ValueError):
            calibration(nst=NUMBER_STOREYS,
                        sdof_capacity=np.array([[0.001, 0.1]]))

    def test_is_sos_non_bool_raises(self):
        with self.assertRaises(TypeError):
            calibration(nst=NUMBER_STOREYS, sdof_capacity=SDOF_CAPACITY,
                        is_sos=1)

    def test_storey_heights_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            calibration(nst=NUMBER_STOREYS, sdof_capacity=SDOF_CAPACITY,
                        storey_heights=[2.8])

    def test_storey_heights_non_positive_raises(self):
        with self.assertRaises(ValueError):
            calibration(nst=NUMBER_STOREYS, sdof_capacity=SDOF_CAPACITY,
                        storey_heights=[0.0, 2.8])

    def test_roof_mass_factor_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            calibration(nst=NUMBER_STOREYS, sdof_capacity=SDOF_CAPACITY,
                        roof_mass_factor=1.5)

    def test_stiffness_group_size_non_int_raises(self):
        with self.assertRaises(TypeError):
            calibration(nst=NUMBER_STOREYS, sdof_capacity=SDOF_CAPACITY,
                        stiffness_group_size=1.5)


if __name__ == "__main__":
    unittest.main()
