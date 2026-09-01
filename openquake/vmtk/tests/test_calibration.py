import unittest
import numpy as np

from openquake.vmtk.calibration import calibrate_model, _dynamic_properties


SDOF_CAPACITY = np.array(
    [[0.00060789, 0.00486316, 0.02420000, 0.04353684],
     [0.10315200, 0.20630401, 0.12378241, 0.12502023]]
).T

NUMBER_STOREYS = 2
STOREY_HEIGHTS = [2.80, 2.80]


class TestCalibrateModel(unittest.TestCase):

    def _calibrate(self, **kwargs):
        defaults = dict(
            nst=NUMBER_STOREYS,
            sdof_capacity=SDOF_CAPACITY,
            is_sos=False,
        )
        defaults.update(kwargs)
        return calibrate_model(**defaults)

    # --- return structure ----------------------------------------------

    def test_returns_five_values(self):
        result = self._calibrate()
        self.assertEqual(len(result), 5)

    def test_floor_masses_length(self):
        floor_masses, *_ = self._calibrate()
        self.assertEqual(len(floor_masses), NUMBER_STOREYS)

    def test_phi_roof_normalised(self):
        _, _, _, phi, _ = self._calibrate()
        self.assertAlmostEqual(phi[-1], 1.0, places=10)

    def test_phi_length(self):
        _, _, _, phi, _ = self._calibrate()
        self.assertEqual(len(phi), NUMBER_STOREYS)

    def test_output_shapes(self):
        _, storey_drifts, storey_forces, _, _ = self._calibrate()
        n_pts = SDOF_CAPACITY.shape[0]
        self.assertEqual(storey_drifts.shape, (NUMBER_STOREYS, n_pts))
        self.assertEqual(storey_forces.shape, (NUMBER_STOREYS, n_pts))

    # --- construction invariant ------------------------------------------

    def test_unit_effective_modal_mass(self):
        floor_masses, _, _, phi, metadata = self._calibrate()
        m_eff = metadata["gamma_real"] * float(np.sum(np.array(floor_masses) * phi))
        self.assertAlmostEqual(m_eff, 1.0, places=8)

    # --- mode shape selection --------------------------------------------

    def test_is_frame_power_law_shape(self):
        phi, _, _ = _dynamic_properties(NUMBER_STOREYS, is_sos=False, is_frame=True)
        expected = np.array(
            [((i + 1) / NUMBER_STOREYS) ** 0.6 for i in range(NUMBER_STOREYS)]
        )
        np.testing.assert_allclose(phi, expected)

    def test_is_frame_ignored_when_soft_storey(self):
        phi_frame_sos, _, _ = _dynamic_properties(
            NUMBER_STOREYS, is_sos=True, is_frame=True
        )
        phi_sos, _, _ = _dynamic_properties(
            NUMBER_STOREYS, is_sos=True, is_frame=False
        )
        np.testing.assert_allclose(phi_frame_sos, phi_sos)

    def test_soft_storey_changes_mode_shape(self):
        phi_sos, _, _ = _dynamic_properties(
            NUMBER_STOREYS, is_sos=True, is_frame=False
        )
        phi_no_sos, _, _ = _dynamic_properties(
            NUMBER_STOREYS, is_sos=False, is_frame=False
        )
        self.assertFalse(np.allclose(phi_sos, phi_no_sos))

    # --- storey_heights pass-through -------------------------------------

    def test_storey_heights_metadata_passthrough(self):
        _, _, _, _, metadata = self._calibrate(storey_heights=STOREY_HEIGHTS)
        self.assertEqual(metadata["storey_heights"], STOREY_HEIGHTS)

    def test_storey_heights_do_not_affect_results(self):
        floor_masses_a, drifts_a, forces_a, phi_a, _ = self._calibrate()
        floor_masses_b, drifts_b, forces_b, phi_b, _ = self._calibrate(
            storey_heights=STOREY_HEIGHTS
        )
        self.assertEqual(floor_masses_a, floor_masses_b)
        np.testing.assert_allclose(drifts_a, drifts_b)
        np.testing.assert_allclose(forces_a, forces_b)
        np.testing.assert_allclose(phi_a, phi_b)

    # --- capacity curve behaviour -----------------------------------------

    def test_storey_drifts_positive_and_increasing(self):
        _, storey_drifts, _, _, _ = self._calibrate()
        for row in storey_drifts:
            self.assertTrue(np.all(row > 0))
            self.assertTrue(np.all(np.diff(row) > 0))

    def test_storey_forces_positive(self):
        _, _, storey_forces, _, _ = self._calibrate()
        self.assertTrue(np.all(storey_forces > 0))

    # --- input validation --------------------------------------------------

    def test_invalid_nst_raises(self):
        with self.assertRaises(ValueError):
            self._calibrate(nst=0)

    def test_nst_float_raises(self):
        with self.assertRaises(ValueError):
            self._calibrate(nst=2.0)

    def test_sdof_capacity_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            self._calibrate(sdof_capacity=np.array([0.001, 0.1]))

    def test_sdof_capacity_too_few_points_raises(self):
        with self.assertRaises(ValueError):
            self._calibrate(sdof_capacity=np.array([[0.001, 0.1]]))

    def test_sdof_capacity_negative_value_raises(self):
        with self.assertRaises(ValueError):
            self._calibrate(
                sdof_capacity=np.array([[0.0, 0.0], [-0.01, 0.1]])
            )

    def test_sdof_capacity_leading_origin_point_is_valid(self):
        result = self._calibrate(
            sdof_capacity=np.array([[0.0, 0.0], [0.01, 0.1], [0.02, 0.15]])
        )
        self.assertEqual(len(result), 5)

    def test_is_sos_non_bool_raises(self):
        with self.assertRaises(TypeError):
            self._calibrate(is_sos=1)

    def test_is_frame_non_bool_raises(self):
        with self.assertRaises(TypeError):
            self._calibrate(is_frame=1)

    def test_storey_heights_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            self._calibrate(storey_heights=[2.8])

    def test_storey_heights_non_positive_raises(self):
        with self.assertRaises(ValueError):
            self._calibrate(storey_heights=[0.0, 2.8])


if __name__ == "__main__":
    unittest.main()
