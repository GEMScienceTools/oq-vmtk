Spectral Acceleration
=====================

.. automethod:: openquake.vmtk.imcalculator.imcalculator.get_sa

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.imcalculator import imcalculator

      acc = np.loadtxt("openquake/vmtk/tests/test_data/acceleration.txt")
      im = imcalculator(acc, dt=0.005)

      sa_1s  = im.get_sa(period=1.0)
      sa_03s = im.get_sa(period=0.3)
      print(f"Sa(1.0s) = {sa_1s:.4f} g,  Sa(0.3s) = {sa_03s:.4f} g")
