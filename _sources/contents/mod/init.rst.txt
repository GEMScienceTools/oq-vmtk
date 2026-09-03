Initialisation
==============

.. automethod:: openquake.vmtk.modeller.modeller.__init__

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.modeller import modeller

      m = modeller(
          number_storeys=2,
          storey_heights=[3.0, 3.0],
          floor_masses=[120.0, 100.0],
          storey_drifts=np.array([[0., 0.005, 0.02, 0.06],
                                  [0., 0.005, 0.02, 0.06]]),
          storey_forces=np.array([[0., 250., 320., 200.],
                                  [0., 220., 280., 180.]]),
          degradation=True,
          # Optional: override individual Pinching4 parameters; any
          # key not provided falls back to the toolkit's default.
          pinching4_params={'gK2': 0.15, 'gFLim': 0.8},
      )
