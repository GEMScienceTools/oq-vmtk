User-Defined Average Spectral Acceleration
==========================================

.. automethod:: openquake.vmtk.imcalculator.imcalculator.get_saavg_user_defined

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.imcalculator import imcalculator

      acc = np.loadtxt("openquake/vmtk/tests/test_data/acceleration.txt")
      im = imcalculator(acc, dt=0.005)

      periods = np.linspace(0.1, 1.0, 10)
      avg_sa = im.get_saavg_user_defined(periods=periods)
      print(f"User-defined AvgSa = {avg_sa:.4f} g")
