Velocity and Displacement History
=================================

.. automethod:: openquake.vmtk.imcalculator.imcalculator.get_velocity_displacement_history

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.imcalculator import imcalculator

      acc = np.loadtxt("openquake/vmtk/tests/test_data/acceleration.txt")
      im = imcalculator(acc, dt=0.005)

      vel, disp = im.get_velocity_displacement_history()
      pgv = max(abs(vel))
      pgd = max(abs(disp))
      print(f"PGV = {pgv:.4f} m/s,  PGD = {pgd:.4f} m")
