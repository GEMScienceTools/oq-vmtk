Initialisation
==============

.. automethod:: openquake.vmtk.slfgenerator.slfgenerator.__init__

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      import pandas as pd
      from openquake.vmtk.slfgenerator import slfgenerator

      inventory = pd.read_csv("demos/StoreyLossFunctionGeneration/in/inventory_psd.csv")
      model = slfgenerator(
          component_data=inventory,
          edp="PSD",
          edp_range=np.linspace(0.001, 0.10, 100),
          grouping_flag=True,
          conversion=1.0,
          realizations=500,
          replacement_cost=1.0,
      )
