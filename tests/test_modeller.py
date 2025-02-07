import os
import sys
import unittest
import numpy as np
from unittest.mock import patch 

vulnerability_toolkit = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(vulnerability_toolkit)
os.chdir(os.path.join(vulnerability_toolkit,'src'))

# Import the modeller class
from src.modeller import modeller

class TestModeller(unittest.TestCase):
    
    def setUp(self):
        """ Setup a modeller instance before each test """
        self.num_storeys   = 3
        self.floor_heights = [3.0, 3.0, 3.0]
        self.floor_masses  = [1000, 1200, 1400]
        self.storey_disps  = np.array([[0.01, 0.02],[0.015, 0.03], [0.02, 0.04]])
        self.storey_forces = np.array([[100,200], [150,300], [200,400]])
        
        self.model = modeller(self.num_storeys, self.floor_heights, self.floor_masses,
                              self.storey_disps, self.storey_forces)
    
    def test_initialisation(self):
        """ Test that the modeller initializes correctly."""
        self.assertEqual(self.model.number_storeys, self.num_storeys)
        self.assertEqual(self.model.floor_heights, self.floor_heights)
        self.assertEqual(self.model.floor_masses, self.floor_masses)
        np.testing.assert_array_equal(self.model.storey_disps, self.storey_disps)
        np.testing.assert_array_equal(self.model.storey_forces, self.storey_forces)
        
    def test_create_pinching4_material(self):
        """Test that Pinching4 material creation runs without errors."""
        with patch("opensees.py.opensees.uniaxialMaterial") as mock_material:
            self.model.create_Pinching4_material(1, 2, self.storey_forces[0], self.storey_disps[0])
            mock_material.assert_called()
        
    def test_compile_model(self):
        """Test that compile_model runs without errors."""
        with patch("openseespy.opensees.model") as mock_model:
            self.model.compile_model()
            mock_model.assert_called()
    
    def test_do_modal_analysis(self):
        "Test modal analysis returns expected number of modes."""
        with patch("openseespy.opensees.eigen", return_value=[1.0, 4.0, 9.0]):
            T, mode_shape = self.model_do_modal_analysis(num_modes=3)
            self.assertEqual(len(T),3)
            self.assertEqual(len(mode_shape), self.number_storeys)
    
    def test_do_spo_analysis(self):
        """Test static pushover analysis runs without crashing."""
        with patch("opensees.py.opensees.analyze", return_value=0):
            result=self.model.do_spo_analysis(0.01, 10, 1, [0.5, 0.75, 1.0])
            self.assertEqual(len(result),4) # Expecting four outputs
    
    def test_do_cpo_analysis(self):
        """Test cyclic pushover analysis runs successfully."""
        with patch("openseespy.opensees.analyze", return_value=0):
            result = self.model.do_cpo_analysis(0.01, 2, 5, 1, 5)
            self.assertEqual(len(result),2)

    def test_do_nrha_analysis(self):
        """Test NRHA analysis logic by mocking OpenSees and file I/O."""
        fake_accel_data = np.array([0.1, 0.2, 0.3, 0.4]) # Simulated acceleration values 
        with patch("openseespy.opensees.analyze", return_value =0) as mock_analyze, \
            patch("openseespy.opensees.getNodeTags", return_value=[1,2,3]) as mock_nodes, \
            patch("openseespy.opensees.nodeDisp", return_value=0.01) as mock_disp,\
            patch("openseespy.opensees.nodeReaction", return_value=-100) as mock_reaction,\
            patch("numpy.loadtxt", return_value=fake_accel_data) as mock_loadtxt,\
            patch("os.remove") as mock_remove: # Mock file deletion
            
            fnames = ["gm_x.txt", "gm_y.txt"] # Fake ground motion files
            dt_gm, sf, t_max, dt_ansys = 0.01, 9.81, 10.0, 0.01
            nrha_outdir = "test_output"
            
            results = self.model.do_nrha_analysis(fnames, dt_gm, sf, t_max, dt_ansys, nrha_outdir)
            
            self.assertEqual(len(results),11)
            
            control_nodes, conv_index, peak_drift, peak_accel, max_peak_drift, max_peak_drift_dir, \
            max_drift_loc, max_peak_accel, max_peak_accel_dir, max_peak_accel_loc, peak_disp  = results
            
            self.assertEqual(conv_index,0) # Expecting stable analysis (0 means no non-convergence instances)
            self.assertIsInstance(peak_drift, np.ndarray)
            self.assertIsInstance(peak_accel, np.ndarray)
            
            # Check if OpenSees functions were called 
            mock_analyze.assert_called()
            mock_nodes.assert_called()
            mock_disp.assert_called()
            mock_reaction.assert_called()
            mock_loadtxt.assert_called()
            
if __name__ == '__main__':
    unittest.main()



