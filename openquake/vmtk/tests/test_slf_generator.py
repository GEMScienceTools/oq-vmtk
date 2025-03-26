import os
import unittest
import pandas as pd

from openquake.vmtk.slf_generator import slf_generator

class TestSLFGenerator(unittest.TestCase):

    def setUp(self):
        """
        Set up the slf_generator instance for each test.
        """
        # Load basicstorey inventory
        cd = os.path.dirname(__file__)
        
        # Define model parameters
        self.slf_file = pd.read_csv(os.path.join(cd, 'test_data', 'slf_inventory.csv'))
        self.regF = 'gpd'
        self.edp = 'PSD'
        self.rlz = 100
        self.grouping_flag=True
        self.conversion = 1.00
        self.repCost = 1.00 
        
        self.model = slf_generator(self.slf_file,
                                   edp= self.edp,
                                   grouping_flag= self.grouping_flag,
                                   conversion = self.conversion,
                                   realizations = self.rlz,
                                   replacement_cost = self.repCost,
                                   regression = self.regF)
        
        
    def test_generate(self):        
        out, cache = self.model.generate()
        self.assertIsNotNone(out)  
        self.assertIsNotNone(cache)

if __name__ == '__main__':
    unittest.main()