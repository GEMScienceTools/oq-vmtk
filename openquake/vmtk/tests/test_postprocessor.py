import os
import unittest
import numpy as np
from scipy import stats
from openquake.vmtk.postprocessor import postprocessor

class TestPostprocessor(unittest.TestCase):

    # Expected values
    b0_value = -6.650269
    b1_value = 2.564951
    sigma    = 0.507748
    tolerance = 0.15  # 10% tolerance
    iml_test = 0.50
    poe_test = 0.50

    def setUp(self):
        """Set up test case with expected regression analysis values."""

        self.pp = postprocessor()
        cd = os.path.dirname(__file__)
        self.imls = np.loadtxt(os.path.join(cd, 'test_data', 'imls.csv'), delimiter=',', usecols=0).tolist()
        self.edps = np.loadtxt(os.path.join(cd, 'test_data', 'edps.csv'), delimiter=',', usecols=0).tolist()
        self.damage_thresholds = [0.003, 0.00992, 0.01708, 0.024]
        self.lower_limit = 0.1 * 0.003
        self.censored_limit = 1.5 * 0.024
        self.sigma_build2build = 0.30
        self.consequence_model = [1.0]

    def within_tolerance(self, expected, actual):
        """Check if actual is within ±10% of expected using np.isclose."""
        if actual is None or np.isnan(actual):
            print(f"Error: Actual value is None or NaN (Expected: {expected})")
            return False

        is_close = np.isclose(actual, expected, rtol=self.tolerance)

        print(f"Expected: {expected}, Actual: {actual}, "
              f"Relative Tolerance: {self.tolerance*100}%, "
              f"Pass: {is_close}")  # Debugging print

        return is_close

    def test_do_cloud_analysis(self):
        """For this test, we try to reproduce the regression analysis of an ESRM20 building class "CR_LDUAL-DUH_H1" and for a single
        intensity measure type (PGA). The ESRM20 analysis outputs are available on
        https://gitlab.seismo.ethz.ch/efehr/esrm20_vulnerability/"""
        cloud_dict = self.pp.do_cloud_analysis(self.imls,
                                               self.edps,
                                               self.damage_thresholds,
                                               self.lower_limit,
                                               self.censored_limit,
                                               sigma_build2build=self.sigma_build2build)

        # Check if keys exist
        for key in ['b0', 'b1', 'sigma']:
            self.assertIn(key, cloud_dict, f"Key '{key}' not found in cloud_dict")

        # Assert values within tolerance
        self.assertTrue(self.within_tolerance(self.b0_value, cloud_dict['b0']),
                        f"b0 value {cloud_dict['b0']} is out of tolerance range for expected {self.b0_value}")

        self.assertTrue(self.within_tolerance(self.b1_value, cloud_dict['b1']),
                        f"b1 value {cloud_dict['b1']} is out of tolerance range for expected {self.b1_value}")

        self.assertTrue(self.within_tolerance(self.sigma, cloud_dict['sigma']),
                        f"sigma value {cloud_dict['sigma']} is out of tolerance range for expected {self.sigma}")
    
    def test_get_fragility_function(self):
        """Check if the probability of exceeding 0.5g is equal to 50% if theta=0.5g"""
        self.assertAlmostEqual(self.pp.get_fragility_function(0.50, 0.40, 0.20, self.iml_test), self.poe_test, places=4)
    
    
    def test_get_rotated_fragility_function(self):
        """Check if a given percentile is equal between the non-rotated and rotated fragility functions"""
        
        theta= 0.50
        beta_r = 0.40
        sigma_build2build = 0.20 
        percentile= 0.10 # We rotate about the median to avoid mismatches due to the interp method
        
        x_values = np.linspace(0.01, 2, 500)
   
        non_rotated_cdf = self.pp.get_fragility_function(theta, 
                                                         beta_r, 
                                                         sigma_build2build=0.00,
                                                         intensities= x_values)
        
        rotated_cdf     = self.pp.get_rotated_fragility_function(theta, 
                                                                 percentile, 
                                                                 beta_r, 
                                                                 sigma_build2build=0.30,
                                                                 intensities = x_values) # Rotated around the 10-th percentile and without the added uncertainty
        
        a = np.interp(percentile, non_rotated_cdf, x_values)   # Get the 10-th percentile from the non-rotated fragility function
        b = np.interp(percentile, rotated_cdf, x_values)       # Get the 10-th percentile from the rotated fragility function
        self.assertAlmostEqual(a, b, places = 5)
        
    def test_get_vulnerability_function(self):

        self.poes = self.pp.get_fragility_function(0.50, 0.30)
        self.poes = np.expand_dims(self.poes, axis=1)
        self.vulnerability = self.pp.get_vulnerability_function(self.poes, self.consequence_model)
        self.assertTrue(
            np.allclose(self.vulnerability['Loss'].values[-1],self.poes[-1],atol=10**-3  # Equivalent to `places=2`
),
            "Arrays are not approximately equal up to 2 decimal places."
        )

if __name__ == '__main__':
    unittest.main()
