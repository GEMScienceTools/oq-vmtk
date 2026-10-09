Amplitude Intensity Measures
============================

.. automethod:: openquake.vmtk.imcalculator.imcalculator.get_amplitude_ims

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.imcalculator import imcalculator

      acc = np.loadtxt("openquake/vmtk/tests/test_data/acceleration.txt")
      im = imcalculator(acc, dt=0.005)

      pga, pgv, pgd = im.get_amplitude_ims()
      print(f"PGA = {pga:.4f} g,  PGV = {pgv:.4f} m/s,  PGD = {pgd:.4f} m")
