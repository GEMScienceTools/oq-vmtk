Nonlinear Time-History Analysis
===============================

.. automethod:: openquake.vmtk.modeller.modeller.do_nrha_analysis

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np

      acc = np.loadtxt("openquake/vmtk/tests/test_data/acceleration.txt")
      m.compile_model()
      m.do_gravity_analysis()
      nrha_results = m.do_nrha_analysis(
          acc=acc,
          dt=0.005,
          xi=0.05,
          dc=0.10,
      )
