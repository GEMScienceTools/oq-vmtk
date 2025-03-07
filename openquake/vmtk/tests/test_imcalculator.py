import os
import unittest
import numpy as np

from openquake.vmtk.im_calculator import IMCalculator

class TestIMCalculator(unittest.TestCase):

    # Test values
    pga_test = 0.54557
    pgv_test = 0.42661
    pgd_test = 0.03304
    sa03_test = 1.26963
    sa06_test = 0.78053
    sa10_test = 0.31027
    avgsa03_test = 1.19491
    avgsa06_test = 0.82186
    avgsa10_test = 0.43333
    periods_list = np.linspace(0.1, 1, 10)
    user_avgsa_test = 0.753149
    ai_test = 0.02069
    cav_test = 1.02289
    t595_test = 7.695
    FIV3_test = 0.073900

    def setUp(self):
        """
        Set up the IMCalculator instance for each test.
        """
        # Load acceleration data and dt from the file
        cd = os.path.dirname(__file__)
        acc_test = np.loadtxt(os.path.join(cd, 'test_data', 'acceleration.txt'))
        dt_test = 0.005

        # Create the IMCalculator object
        self.calculator = IMCalculator(acc_test, dt_test)

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
        sa_avg_user = self.calculator.get_saavg_user_defined(self.periods_list)
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

    def test_get_FIV3(self):
        FIV3, _, _, _, _, _ = self.calculator.get_FIV3(period=0.3, alpha=1.0, beta=0.7)
        self.assertAlmostEqual(FIV3, self.FIV3_test, places=3)

if __name__ == '__main__':
    unittest.main()
