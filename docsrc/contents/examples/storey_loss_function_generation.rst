Storey Loss Function Generation
===============================

Demonstrates how to generate Storey Loss Functions (SLFs) using the
``slfgenerator`` class. Covers the FEMA P-58 component inventory (sequential,
mutually exclusive and simultaneous damage states), the exact expected storey loss
ratio, its Monte Carlo dispersion and 16th/50th/84th percentiles, and optional
correlation trees between dependent components.

The full notebook is at
`demos/StoreyLossFunctionGeneration/StoreyLossFunctionGeneration.ipynb
<https://github.com/GEMScienceTools/oq-vmtk/blob/incremental-dynamic-analysis/demos/StoreyLossFunctionGeneration/StoreyLossFunctionGeneration.ipynb>`_.
The required inventory CSVs are bundled at
`demos/StoreyLossFunctionGeneration/in/
<https://github.com/GEMScienceTools/oq-vmtk/tree/main/demos/StoreyLossFunctionGeneration/in>`_.

The snippet below reproduces the same workflow and can be run from the
repository root after ``pip install .``.

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

   # 2. Common generator settings. Each group's loss is divided by the
   #    replacement cost of its own components (the default), so "slf" is the
   #    expected storey loss ratio.
   common = dict(
       grouping_flag=True,
       conversion=1.0,
       realizations=2000,
       seed=2026,
   )

   # 3. Drift-sensitive SLF
   psd_model = slfgenerator(
       component_data=inventory_psd,
       edp="PSD",
       edp_range=np.linspace(0.0, 0.10, 101),
       **common,
   )
   psd_slf, psd_cache = psd_model.generate()
   export_to_pkl("out/slf_drift.pkl", psd_slf)

   # 4. Acceleration-sensitive SLF
   pfa_model = slfgenerator(
       component_data=inventory_pfa,
       edp="PFA",
       edp_range=np.linspace(0.0, 5.0, 101),
       **common,
   )
   pfa_slf, pfa_cache = pfa_model.generate()
   export_to_pkl("out/slf_accel.pkl", pfa_slf)

   # 5. Visualise both
   pl = plotter()
   pl.plot_slf_model(
       psd_slf, psd_cache,
       edp_label="Interstorey Drift Ratio [-]",
       loss_label="Drift-Sensitive NSC Storey Loss Ratio",
       xlims=[0, 0.05], ylims=[0, 1], title="Drift-Sensitive SLF",
   )
   pl.plot_slf_model(
       pfa_slf, pfa_cache,
       edp_label="Peak Floor Acceleration [g]",
       loss_label="Acceleration-Sensitive NSC Storey Loss Ratio",
       xlims=[0, 5.0], ylims=[0, 1], title="Acceleration-Sensitive SLF",
   )

   # 6. Combine the two groups into the nonstructural loss ratio of a storey,
   #    weighting each by its share of the replacement cost
   rc_d = psd_slf["PSD, NS"]["replacement_cost"]
   rc_a = pfa_slf["PFA, NS"]["replacement_cost"]
   w_d = rc_d / (rc_d + rc_a)
   psd, pfa = 0.01, 0.5        # storey drift and floor acceleration
   storey_loss = (w_d * np.interp(psd, psd_slf["PSD, NS"]["edp_range"], psd_slf["PSD, NS"]["slf"])
                  + (1 - w_d) * np.interp(pfa, pfa_slf["PFA, NS"]["edp_range"], pfa_slf["PFA, NS"]["slf"]))
