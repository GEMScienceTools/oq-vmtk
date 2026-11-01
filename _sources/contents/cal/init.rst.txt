Initialisation
==============

.. automethod:: openquake.vmtk.calibration.calibration.__init__

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.calibration import calibration

      sdof_capacity = np.array([
          [0.000, 0.00],
          [0.020, 0.18],
          [0.080, 0.22],
          [0.150, 0.10],
      ])
      cal = calibration(
          nst=4,
          sdof_capacity=sdof_capacity,
          storey_heights=[3.0, 3.0, 3.0, 3.0],
          roof_mass_factor=0.75,
      )
