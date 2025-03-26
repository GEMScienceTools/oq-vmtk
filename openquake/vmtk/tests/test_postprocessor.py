import os
import unittest
import numpy as np
from scipy import stats
from openquake.vmtk.postprocessor import postprocessor

class TestPostprocessor(unittest.TestCase):

    # Expected values
    # These values correspond to the regression coefficients of the "CR_LDUAL-DUH_H1"
    # building class from ESRM20 accessible here: 
    # https://gitlab.seismo.ethz.ch/efehr/esrm20_vulnerability/-/blob/master/scripts/vmtk/outputs/regression/CR_LDUAL-DUH_H1/regression_coeff_PGA.csv?ref_type=heads
    
    b0_value = -6.650269
    b1_value = 2.564951
    sigma    = 0.507748
    tolerance = 0.01  # 1% tolerance
    iml_test = 0.50
    poe_test = 0.50
    
    def setUp(self):
        """Set up test case with expected regression analysis values."""

        self.pp = postprocessor()
        cd = os.path.dirname(__file__)
        self.cap_array = np.array(([0.0, 0.0],[0.0005313, 1.025], [0.004, 2.05], [0.024, 2.071]))
        self.imls = np.loadtxt(os.path.join(cd, 'test_data', 'imls.csv'), delimiter=',', usecols=0).tolist()
        self.edps = np.loadtxt(os.path.join(cd, 'test_data', 'edps.csv'), delimiter=',', usecols=0).tolist()
        
        if self.cap_array.shape[0]>=4:
            self.dam_model_drift=[0.75*self.cap_array[2,0],\
                             0.5*self.cap_array[2,0]+0.33*self.cap_array[-1,0],\
                             0.25*self.cap_array[2,0]+0.67*self.cap_array[-1,0],\
                             self.cap_array[-1,0]]
            self.low_disp_limit = 0.1*self.cap_array[2,0]
        else:
            self.dam_model_drift=[0.75*self.cap_array[1,0],\
                             0.5*self.cap_array[1,0]+0.33*self.cap_array[-1,0],\
                             0.25*self.cap_array[1,0]+0.67*self.cap_array[-1,0],\
                             self.cap_array[-1,0]]
            self.low_disp_limit = 0.1*self.cap_array[1,0]
        self.censored_limit = 1.5 * self.cap_array[-1,0]
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
                                               self.dam_model_drift,
                                               self.low_disp_limit,
                                               self.censored_limit,
                                               sigma_build2build=self.sigma_build2build,
                                               fragility_method = 'lognormal')

        # Check if keys exist
        for key in ['cloud inputs', 'fragility', 'regression']:
            self.assertIn(key, cloud_dict, f"Key '{key}' not found in cloud_dict")
        
        # Assert values within tolerance
        self.assertTrue(self.within_tolerance(self.b0_value, cloud_dict['regression']['b0']),
                        f"b0 value {cloud_dict['regression']['b0']} is out of tolerance range for expected {self.b0_value}")

        self.assertTrue(self.within_tolerance(self.b1_value, cloud_dict['regression']['b1']),
                        f"b1 value {cloud_dict['regression']['b1']} is out of tolerance range for expected {self.b1_value}")

        self.assertTrue(self.within_tolerance(self.sigma, cloud_dict['regression']['sigma']),
                        f"sigma value {cloud_dict['regression']['sigma']} is out of tolerance range for expected {self.sigma}")
    
    def test_calculate_lognormal_fragility(self):
        """Check if the probability of exceeding 0.5g is equal to 50% if theta=0.5g"""
        self.assertAlmostEqual(self.pp.calculate_lognormal_fragility(0.50, 0.40, 0.20, self.iml_test), self.poe_test, places=4)
    

    def test_calculate_rotated_fragility(self):
        """Check if a given percentile is equal between the non-rotated and rotated fragility functions"""
        
        theta               = 0.50
        sigma_record2record = 0.40
        percentile          = 0.10 # We rotate about the 10th percentile
        x_values = np.linspace(0.01, 2, 500)
   
        non_rotated_cdf = self.pp.calculate_lognormal_fragility(theta, 
                                                                sigma_record2record, 
                                                                sigma_build2build=0.00,
                                                                intensities= x_values)
        
        _, _, rotated_cdf= self.pp.calculate_rotated_fragility(theta, 
                                                              percentile, 
                                                              sigma_record2record, 
                                                              sigma_build2build=0.00,
                                                              intensities = x_values) # Rotated around the 10-th percentile and without the added uncertainty
        
        a = np.interp(percentile, non_rotated_cdf, x_values)   # Get the 10-th percentile from the non-rotated fragility function
        b = np.interp(percentile, rotated_cdf, x_values)       # Get the 10-th percentile from the rotated fragility function
        self.assertAlmostEqual(a, b, places = 5)
        
    def test_get_vulnerability_function(self):

        self.poes = self.pp.calculate_lognormal_fragility(0.50, 0.30)
        self.poes = np.expand_dims(self.poes, axis=1)
        self.vulnerability = self.pp.get_vulnerability_function(self.poes, self.consequence_model)
        self.assertTrue(
            np.allclose(self.vulnerability['Loss'].values[-1],self.poes[-1],atol=10**-3  # Equivalent to `places=2`
),
            "Arrays are not approximately equal up to 2 decimal places."
        )

if __name__ == '__main__':
    unittest.main()
