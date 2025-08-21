import os
import shutil
import unittest
import numpy as np

from openquake.vmtk.calibration import calibrate_model
from openquake.vmtk.modeller import modeller

class TestModeller(unittest.TestCase):

    def setUp(self):
        """
        Set up the modeller instance for each test.
        """
        cd = os.path.dirname(__file__)
        self.number_storeys = 2
        self.storey_heights = [2.80, 2.80]
        self.gamma = 1.33
        self.damping = 0.05
        self.degradation = True
        self.sdof_capacity = np.array([[0.00060789, 0.00486316, 0.02420000, 0.04353684],
                                       [0.10315200, 0.20630401, 0.12378241, 0.12502023]]).T
        self.isSOS = False
        self.isFrame = False
        self.fnames = [os.path.join(cd, 'test_data', 'acceleration.txt')]
        self.dt_gm = 0.005
        self.t_max = 30.0
        self.temp_folder = os.path.join(cd, 'test_data', 'temp')
        os.makedirs(self.temp_folder, exist_ok=True)

        # Expected results
        self.masses_test = [0.5979496788391293, 0.448462259129347]
        self.disps_test = np.array([[0.00052664, 0.00421318, 0.02096557, 0.03771796],
                                    [0.00028185, 0.00225482, 0.01122043, 0.02018604]])
        self.forces_test = np.array([[0.11496146, 0.22992293, 0.13795376, 0.1393333 ],
                                     [0.06152551, 0.12305102, 0.07383061, 0.07456892]])
        self.phi_test = np.array([0.65138782, 1.])
        self.T_test= 0.154

        # Initialize the model here so that it is ready for all tests
        self.model = modeller(self.number_storeys,
                              self.storey_heights,
                              self.masses_test,
                              self.disps_test,
                              self.forces_test * 9.81,  # Apply gravity
                              self.degradation)  # Initialize the model

        self.model.compile_model()  # Compile the model in the setUp method

    def test_calibrate(self):
        masses, disps, forces, phi = calibrate_model(self.number_storeys,
                                                     self.gamma,
                                                     self.sdof_capacity,
                                                     self.isFrame,
                                                     self.isSOS)

        self.assertEqual(masses, self.masses_test)
        np.testing.assert_allclose(disps, self.disps_test, atol=1e-8)  # Specify absolute tolerance
        np.testing.assert_allclose(forces, self.forces_test, atol=1e-8)
        np.testing.assert_allclose(phi, self.phi_test, atol=1e-8)

    def test_modeller(self):
        # The model is already initialized in setUp, so no need to reinitialize it
        self.model.compile_model()

    def test_gravity_analysis(self):
        # The model is initialized in setUp, so we can directly call this method
        self.model.do_gravity_analysis()

    def test_modal_analysis(self):

        num_modes = 3
        T, phi = self.model.do_modal_analysis(num_modes=num_modes)  # Do modal analysis and get period of vibration
        self.assertAlmostEqual(T[0], self.T_test, places= 4)
        self.assertAlmostEqual(phi[0], self.phi_test[0], places=4)

    def test_do_spo_analysis(self):

        self.model.do_spo_analysis(0.01, 5, 1, self.phi_test, pflag=False)

    def test_do_cpo_analysis(self):

        self.model.do_cpo_analysis(0.01, [1,5,10], 1, 2, self.phi_test, pflag=False)
        
    def test_do_nrha_analysis(self):

        num_modes = 3
        T, phi = self.model.do_modal_analysis(num_modes=num_modes)  # Do modal analysis and get period of vibration (necessary for nltha)
        self.model.do_nrha_analysis(self.fnames,
                                    self.dt_gm,
                                    9.81,
                                    self.t_max,
                                    0.001,
                                    self.temp_folder,
                                    pflag=False,
                                    xi = self.damping)
        # Remove temporary folder
        shutil.rmtree(self.temp_folder)

if __name__ == '__main__':
    unittest.main()
