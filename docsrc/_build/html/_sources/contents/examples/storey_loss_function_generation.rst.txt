Storey Loss Function Generation
===============================

Demonstrates how to generate Storey Loss Functions (SLFs) using the
``slfgenerator`` class. Covers component inventory input, Monte Carlo
simulation of damage states, and regression fitting of SLF curves.

The full notebook is at
`demos/StoreyLossFunctionGeneration/StoreyLossFunctionGeneration.ipynb
<https://github.com/GEMScienceTools/oq-vmtk/blob/incremental-dynamic-analysis/demos/StoreyLossFunctionGeneration/StoreyLossFunctionGeneration.ipynb>`_.
The required inventory CSVs are bundled at
`demos/StoreyLossFunctionGeneration/in/
<https://github.com/GEMScienceTools/oq-vmtk/tree/main/demos/StoreyLossFunctionGeneration/in>`_.

Self-contained code
-------------------

Run the snippet below from the repository root after ``pip install .``.
It is the same workflow as the notebook, condensed into a single
copy-pasteable script.

.. code-block:: python

   import os
   import numpy as np
   import pandas as pd

   from openquake.vmtk.slfgenerator import slfgenerator
   from openquake.vmtk.plotter import plotter
   from openquake.vmtk.utilities import export_to_pkl

   inventory_dir = "demos/StoreyLossFunctionGeneration/in"
   os.makedirs("out", exist_ok=True)

   # 1. Load the bundled FEMA P-58 component inventories
   inventory_psd = pd.read_csv(os.path.join(inventory_dir, "inventory_psd.csv"))
   inventory_pfa = pd.read_csv(os.path.join(inventory_dir, "inventory_pfa.csv"))

   # 2. Common generator settings
   common = dict(
       grouping_flag=True,
       conversion=1.0,
       realizations=500,
       replacement_cost=1.0,
       regression=None,        # auto-select the best fit
   )

   # 3. Drift-sensitive SLF
   psd_model = slfgenerator(
       component_data=inventory_psd,
       edp="PSD",
       edp_range=np.linspace(0.001, 0.10, 100),
       **common,
   )
   psd_slf, psd_cache = psd_model.generate()
   export_to_pkl("out/slf_drift.pkl", psd_slf)

   # 4. Acceleration-sensitive SLF
   pfa_model = slfgenerator(
       component_data=inventory_pfa,
       edp="PFA",
       edp_range=np.linspace(0.001, 5.0, 100),
       **common,
   )
   pfa_slf, pfa_cache = pfa_model.generate()
   export_to_pkl("out/slf_accel.pkl", pfa_slf)

   # 5. Visualise both
   pl = plotter()
   pl.plot_slf_model(
       psd_slf, psd_cache,
       edp_label="Interstorey Drift Ratio [-]",
       loss_label="Drift-Sensitive NSC Storey Loss",
       xlims=[0, 0.05], title="Drift-Sensitive SLF",
   )
   pl.plot_slf_model(
       pfa_slf, pfa_cache,
       edp_label="Peak Floor Acceleration [g]",
       loss_label="Acceleration-Sensitive NSC Storey Loss",
       xlims=[0, 5.0], title="Acceleration-Sensitive SLF",
   )
