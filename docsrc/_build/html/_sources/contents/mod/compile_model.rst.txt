Model Compilation
=================

.. automethod:: openquake.vmtk.modeller.modeller.compile_model

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.modeller import modeller

      m = modeller(
          number_storeys=2, storey_heights=[3.0, 3.0],
          floor_masses=[120.0, 100.0],
          storey_drifts=np.array([[0., 0.005, 0.02, 0.06],
                                  [0., 0.005, 0.02, 0.06]]),
          storey_forces=np.array([[0., 250., 320., 200.],
                                  [0., 220., 280., 180.]]),
      )
      m.compile_model()
