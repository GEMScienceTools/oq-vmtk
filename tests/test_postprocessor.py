import os
import sys
import unittest
import numpy as np
import pandas as pd

vulnerability_toolkit = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(vulnerability_toolkit)
os.chdir(os.path.join(vulnerability_toolkit,'src'))

# Import the postprocessor class
from src.postprocessor import postprocessor

class TestPostProcessor(unittest.TestCase):
    
    def setUp(self):
        """Initialise a postprocessor instance before each test."""
        
        self.pp = postprocessor()
        self.imls = np.array([0.1, 0.5, 1.0, 2.0, 5.0])            # Example intensity measures levels
        self.edps = np.array([0.01, 0.05, 0.1, 0.2, 0.5])          # Example engineering demand parameters
        self.damage_thresholds = np.array([0.02, 0.08, 0.15, 0.3]) # Example damage thresholds
        self.consequence_model = [0.05, 0.15, 0.6, 1.0]            # Example damage-to-loss model
        
    def test_do_cloud_analysis(self):
        """Test cloud analysis returns valid structure and values."""
        
        result = self.pp.do_cloud_analysis(
            imls=self.imls,
            edps=self.edps,
            damage_thresholds=self.damage_thresholds,
            lower_limit = 0.01,
            censored_limit = 0.4)
        
        self.assertIsInstance(result, dict)
        self.assertIn("imls", result)
        self.assertIn("poes", result)
        self.assertIn("medians", result)
        self.assertIn("betas_total", result)
        
        # Check correct number of probability of exceedance values
        self.assertEqual(len(result["poes"][0]), len(self.damage_thresholds))
        
    def test_get_fragility_function(self):
        """Test fragility function probabilities are between 0 and 1."""
        fragility = self.pp.get_fragility_function(1.0, 0.3)
        self.assertTrue(np.all(fragility>=0))
        self.assertTrue(np.all(fragility<=1))
        
    def test_get_vulnerability_function(self):
        "Test vulnerability function returns expected loss values."""
        poes = np.array([
            [0.1, 0.2, 0.4, 0.7],
            [0.2, 0.3, 0.5, 0.8],
            [0.3, 0.4, 0.6, 0.9]
            ])
        result = self.pp.get_vulnerability_function(poes, self.consequence_model)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Loss", result.columns)
        self.assertTrue(np.all(result["Loss"]>=0))
        self.assertTrue(np.all(result["Loss"]<=1))
        
        
    def test_calculate_sigma_loss(self):
        """Test loss uncertainty calculation returns reasonable values."""
        loss_values = np.array([0.01, 0.1, 0.5, 0.9, 1.0])
        sigma_loss, _, _ = self.pp.calculate_sigma_loss(loss_values)
        self.assertEqual(len(sigma_loss), len(loss_values))
        self.assertTrue(np.all(sigma_loss>=0))

if __name__ == '__main__':
    unittest.main()
    
    
        
    
        

