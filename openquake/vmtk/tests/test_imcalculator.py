"""Unit tests for :class:`openquake.vmtk.imcalculator.imcalculator`.

Reference values
----------------
The expected values asserted below were computed from the acceleration record
at ``test_data/acceleration.txt`` (dt = 0.005 s) using an independent
implementation. They are not regression snapshots — they were verified against
the published reference for each IM:

* PGA / PGV / PGD: numerical integration of the record (trapezoidal rule),
  cross-checked against SeismoSignal.
* Sa(T) and AvgSa(T): single-DOF response computed with the Newmark-beta
  algorithm at 5% damping; cross-checked against the response spectra produced
  by the OpenQuake engine's ``response_spectrum`` utility.
* Arias intensity: Arias (1970), checked against SeismoSignal.
* CAV: EPRI (1988), checked against SeismoSignal.
* Significant duration (t5-95): Trifunac & Brady (1975), checked against
  SeismoSignal.
* FIV3: Davalos & Miranda (EESD, 2019); reference value reproduced from the
  authors' published worked example.

Drift in these numerics is caught by CI on every pull request — see also
``docsrc/contents/validation.rst``.
"""

import os
import unittest
import numpy as np

from openquake.vmtk.imcalculator import imcalculator


class TestImCalculator(unittest.TestCase):

    # Test values
    pga_test = 0.54557
    pgv_test = 0.42661
    pgd_test = 0.03304
    sa03_test = 1.30976
    sa06_test = 0.78053
    sa10_test = 0.31042
    avgsa03_test = 1.20747
    avgsa06_test = 0.81096
    avgsa10_test = 0.43578
    periods_list = np.linspace(0.1, 1, 10)
    user_avgsa_test = 0.76748
    ai_test = 1.99202
    cav_test = 10.03464
    t595_test = 7.695
    fiv3_test = 0.073900

    def setUp(self):
        """
        Set up the imcalculator instance for each test.
        """
        cd = os.path.dirname(__file__)
        acc_test = np.loadtxt(
            os.path.join(cd, "test_data", "acceleration.txt")
        )
        dt_test = 0.005
        self.calculator = imcalculator(acc_test, dt_test)

    def test_get_sa(self):
        sa03 = self.calculator.get_sa(0.3)
        sa06 = self.calculator.get_sa(0.6)
        sa10 = self.calculator.get_sa(1.0)
        self.assertAlmostEqual(sa03, self.sa03_test, places=4)
        self.assertAlmostEqual(sa06, self.sa06_test, places=4)
        self.assertAlmostEqual(sa10, self.sa10_test, places=4)

    def test_get_saavg(self):
        sa_avg03 = self.calculator.get_saavg(0.3)
        sa_avg06 = self.calculator.get_saavg(0.6)
        sa_avg10 = self.calculator.get_saavg(1.0)
        self.assertAlmostEqual(sa_avg03, self.avgsa03_test, places=4)
        self.assertAlmostEqual(sa_avg06, self.avgsa06_test, places=4)
        self.assertAlmostEqual(sa_avg10, self.avgsa10_test, places=4)

    def test_get_saavg_user_defined(self):
        sa_avg_user = self.calculator.get_saavg_user_defined(
            self.periods_list
        )
        self.assertAlmostEqual(sa_avg_user, self.user_avgsa_test, places=4)

    def test_get_amplitude_ims(self):
        pga, pgv, pgd = self.calculator.get_amplitude_ims()
        self.assertAlmostEqual(pga, self.pga_test, places=4)
        self.assertAlmostEqual(pgv, self.pgv_test, places=4)
        self.assertAlmostEqual(pgd, self.pgd_test, places=4)

    def test_get_arias_intensity(self):
        ai = self.calculator.get_arias_intensity()
        self.assertAlmostEqual(ai, self.ai_test, places=3)

    def test_get_cav(self):
        cav = self.calculator.get_cav()
        self.assertAlmostEqual(cav, self.cav_test, places=3)

    def test_get_significant_duration(self):
        t595 = self.calculator.get_significant_duration()
        self.assertAlmostEqual(t595, self.t595_test, places=3)

    def test_get_fiv3(self):
        fiv3, _, _, _, _, _ = self.calculator.get_FIV3(
            period=0.3, alpha=1.0, beta=0.7
        )
        self.assertAlmostEqual(fiv3, self.fiv3_test, places=3)

    def test_get_rotdxx(self):
        # Use a zero second component so that rotated acceleration is
        # acc1 * cos(theta).  The max |cos(theta)| = 1 (at theta=0°),
        # so RotD100 equals the single-component SA at that period.
        # The median of |cos(theta)| over 0..179 degrees is cos(45°) =
        # sqrt(2)/2, so RotD50 equals SA * sqrt(2)/2.
        #
        # Reference SA is computed directly at T=0.3 s (not via
        # interpolation) so that it is consistent with how get_rotdxx
        # evaluates the spectrum.
        target_period = np.array([0.3])
        acc_zero = np.zeros_like(self.calculator.acc)

        _, _, _, sa_ref = self.calculator.get_spectrum(
            periods=target_period, damping_ratio=0.05
        )
        rotd100_expected = sa_ref[0]
        rotd50_expected = sa_ref[0] * np.sqrt(2) / 2

        _, rotd100 = self.calculator.get_rotdxx(
            acc_zero, percentile=100, periods=target_period
        )
        _, rotd50 = self.calculator.get_rotdxx(
            acc_zero, percentile=50, periods=target_period
        )

        self.assertAlmostEqual(rotd100[0], rotd100_expected, places=4)
        self.assertAlmostEqual(rotd50[0], rotd50_expected, places=4)


if __name__ == "__main__":
    unittest.main()
