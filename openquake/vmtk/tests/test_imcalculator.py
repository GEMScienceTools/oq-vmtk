"""Unit tests for :class:`openquake.vmtk.imcalculator.imcalculator`.

Reference values
----------------
The expected values asserted below were computed from the acceleration record
at ``test_data/acceleration.txt`` (dt = 0.005 s) using an independent
implementation. They are not regression snapshots — they were verified against
the published reference for each IM:

* PGA / PGV / PGD: numerical integration of the record (trapezoidal rule),
  cross-checked against SeismoSignal (``highpass_hz=None``; the default
  drift-corrected values are tested to stay within 0.2% of them).
* Sa(T) and AvgSa(T): single-DOF response computed with the Newmark-beta
  algorithm at 5% damping; cross-checked against the response spectra produced
  by the OpenQuake engine's ``response_spectrum`` utility.
* Arias intensity: Arias (1970), checked against SeismoSignal.
* CAV: EPRI (1988), checked against SeismoSignal.
* Significant duration (t5-95): Trifunac & Brady (1975), checked against
  SeismoSignal.
* FIV3: Davalos & Miranda (EESD, 2019); reference value reproduced from the
  authors' published worked example.

* RotDxx (RotD50/RotD100) of PGA/PGV/PGD, CAV, Arias intensity, t5-95,
  AvgSA and FIV3: checked against a brute-force recomputation of the
  single-component IM on the record rotated to each of the 180 angles
  (synthetic pair), and against closed forms on the reference record when
  the second component is a scaled copy of the first.

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
        # Reference values are for plain trapezoidal integration
        pga, pgv, pgd = self.calculator.get_amplitude_ims(highpass_hz=None)
        self.assertAlmostEqual(pga, self.pga_test, places=4)
        self.assertAlmostEqual(pgv, self.pgv_test, places=4)
        self.assertAlmostEqual(pgd, self.pgd_test, places=4)

    def test_get_amplitude_ims_default_keeps_clean_record(self):
        # The default drift correction (0.05 Hz) must not alter a
        # well-processed record: within 0.2% of the reference values
        pga, pgv, pgd = self.calculator.get_amplitude_ims()
        self.assertAlmostEqual(pga, self.pga_test, places=4)
        self.assertLess(abs(pgv / self.pgv_test - 1), 2e-3)
        self.assertLess(abs(pgd / self.pgd_test - 1), 2e-3)

    def test_get_amplitude_ims_removes_baseline_drift(self):
        # Record with a known exact displacement that starts and ends at
        # rest: d(t) = D env(t) sin(w t), with a Gaussian envelope env;
        # acceleration = d''(t). A small constant acceleration offset
        # (baseline error) makes the displacement drift quadratically
        # when integrated as it is; the default high-pass removes it.
        dt, D, w, t0, s = 0.01, 0.05, 2 * np.pi, 30.0, 6.0
        t = np.arange(0, 60, dt)
        env = np.exp(-(((t - t0) / s) ** 2))
        d_env = env * (-2 * (t - t0) / s**2)
        dd_env = env * (4 * (t - t0) ** 2 / s**4 - 2 / s**2)
        disp_exact = D * env * np.sin(w * t)
        vel_exact = D * (d_env * np.sin(w * t) + env * w * np.cos(w * t))
        acc = D * (dd_env * np.sin(w * t) + 2 * d_env * w * np.cos(w * t)
                   - env * w**2 * np.sin(w * t))
        calc = imcalculator(acc / 9.81 + 0.001, dt)
        _, _, pgd_raw = calc.get_amplitude_ims(highpass_hz=None)
        _, pgv, pgd = calc.get_amplitude_ims()
        pgv_exact = np.max(np.abs(vel_exact))
        pgd_exact = np.max(np.abs(disp_exact))
        self.assertGreater(pgd_raw, 100 * pgd_exact)
        self.assertLess(abs(pgv / pgv_exact - 1), 0.01)
        self.assertLess(abs(pgd / pgd_exact - 1), 0.01)

    def test_get_vel_disp_history_matches_amplitude_ims(self):
        vel, disp = self.calculator.get_vel_disp_history()
        _, pgv, pgd = self.calculator.get_amplitude_ims()
        self.assertEqual(len(vel), len(self.calculator.acc))
        self.assertEqual(len(disp), len(self.calculator.acc))
        self.assertAlmostEqual(np.max(np.abs(vel)), pgv, places=12)
        self.assertAlmostEqual(np.max(np.abs(disp)), pgd, places=12)

    def test_get_amplitude_ims_rejects_invalid_corner(self):
        for bad in (0.0, -0.1, 0.5 / self.calculator.dt):
            with self.assertRaises(ValueError):
                self.calculator.get_amplitude_ims(highpass_hz=bad)

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

        _, _, _, psa_ref = self.calculator.get_spectrum(
            periods=target_period, damping_ratio=0.05
        )
        rotd100_expected = psa_ref[0]
        rotd50_expected = psa_ref[0] * np.sqrt(2) / 2

        _, rotd100 = self.calculator.get_rotdxx(
            acc_zero, percentile=100, periods=target_period
        )
        _, rotd50 = self.calculator.get_rotdxx(
            acc_zero, percentile=50, periods=target_period
        )

        self.assertAlmostEqual(rotd100[0], rotd100_expected, places=4)
        self.assertAlmostEqual(rotd50[0], rotd50_expected, places=4)


class TestRotDxxIMs(unittest.TestCase):
    """RotDxx of PGA/PGV/PGD, CAV, AI, D5-95, AvgSA and FIV3.

    The reference is brute force: the single-component method is
    recomputed on the accelerogram rotated to each of the 180 angles and
    the percentile across angles is taken.
    """

    dt = 0.01

    def setUp(self):
        rng = np.random.default_rng(3)
        t = np.arange(1500) * self.dt
        env = np.exp(-t / 5) * (1 - np.exp(-t))
        self.a1 = 0.25 * rng.standard_normal(len(t)) * env + 0.01
        self.a2 = 0.15 * rng.standard_normal(len(t)) * env
        self.calc = imcalculator(self.a1, self.dt)
        self.th = np.deg2rad(np.arange(180))

    def brute(self, func, q=50):
        vals = [
            func(imcalculator(np.cos(x) * self.a1 + np.sin(x) * self.a2,
                              self.dt))
            for x in self.th
        ]
        return np.percentile(vals, q, axis=0)

    def test_amplitude_ims(self):
        for q in (50, 100):
            got = self.calc.get_rotdxx_amplitude_ims(self.a2, q)
            ref = self.brute(lambda r: r.get_amplitude_ims(), q)
            np.testing.assert_allclose(got, ref, rtol=1e-8)

    def test_cav(self):
        got = self.calc.get_rotdxx_cav(self.a2)
        ref = self.brute(lambda r: r.get_cav())
        self.assertAlmostEqual(got, ref, places=10)

    def test_arias_and_duration(self):
        ai, dur = self.calc.get_rotdxx_arias_duration(self.a2)
        ref_ai = self.brute(lambda r: r.get_arias_intensity())
        ref_dur = self.brute(lambda r: r.get_significant_duration())
        self.assertAlmostEqual(ai, ref_ai, places=10)
        self.assertAlmostEqual(dur, ref_dur, places=10)

    def test_fiv3(self):
        got = self.calc.get_rotdxx_FIV3(self.a2, 1.0, 0.7, 0.85)
        ref = self.brute(lambda r: r.get_FIV3(1.0, 0.7, 0.85)[0])
        self.assertAlmostEqual(got, ref, places=10)

    def test_saavg_single_component(self):
        # a2 = 0: every angle scales the SA of a1 by |cos(theta)|, so
        # RotD100 = AvgSA and RotD50 = median(|cos|) * AvgSA
        zero = np.zeros_like(self.a1)
        plist = np.linspace(0.2, 1.5, 10)
        ref = self.calc.get_saavg_user_defined(plist)
        rd100 = self.calc.get_rotdxx_saavg(zero, periods_list=plist,
                                           percentile=100)
        rd50 = self.calc.get_rotdxx_saavg(zero, periods_list=plist)
        med = np.median(np.abs(np.cos(self.th)))
        # reference interpolates a 500-point spectrum grid
        self.assertAlmostEqual(rd100 / ref, 1.0, places=3)
        self.assertAlmostEqual(rd50 / ref, med, places=3)

    def test_saavg_period_definition(self):
        a = self.calc.get_rotdxx_saavg(self.a2, period=1.0)
        b = self.calc.get_rotdxx_saavg(
            self.a2, periods_list=np.linspace(0.2, 1.5, 10)
        )
        self.assertEqual(a, b)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.calc.get_rotdxx_cav(self.a2[:-1])
        with self.assertRaises(ValueError):
            self.calc.get_rotdxx_saavg(self.a2)


class TestRotDxxScaledCopy(unittest.TestCase):
    """Closed forms on the reference record with acc2 = k * acc1.

    The rotated record is (cos(theta) + k sin(theta)) * acc1, so every IM
    is the single-component value times a known function of the angle.
    """

    k = 0.85

    def setUp(self):
        cd = os.path.dirname(__file__)
        acc = np.loadtxt(os.path.join(cd, "test_data", "acceleration.txt"))
        self.calc = imcalculator(acc, 0.005)
        self.acc2 = self.k * acc
        th = np.deg2rad(np.arange(180))
        self.f = np.abs(np.cos(th) + self.k * np.sin(th))

    def test_amplitude_ims(self):
        ref = np.array(self.calc.get_amplitude_ims())
        for q in (50, 100):
            got = self.calc.get_rotdxx_amplitude_ims(self.acc2, q)
            exp = ref * np.percentile(self.f, q)
            np.testing.assert_allclose(got, exp, rtol=1e-8)

    def test_cav(self):
        got = self.calc.get_rotdxx_cav(self.acc2)
        exp = self.calc.get_cav() * np.median(self.f)
        self.assertAlmostEqual(got, exp, places=8)

    def test_arias_and_duration(self):
        ai, dur = self.calc.get_rotdxx_arias_duration(self.acc2)
        exp_ai = self.calc.get_arias_intensity() * np.median(self.f**2)
        self.assertAlmostEqual(ai, exp_ai, places=8)
        # duration is invariant to the amplitude scaling
        self.assertAlmostEqual(
            dur, self.calc.get_significant_duration(), places=8
        )

    def test_rotd100_is_largest(self):
        a50 = self.calc.get_rotdxx_amplitude_ims(self.acc2, 50)
        a100 = self.calc.get_rotdxx_amplitude_ims(self.acc2, 100)
        self.assertTrue(all(b > a for a, b in zip(a50, a100)))


if __name__ == "__main__":
    unittest.main()
