Demand Profiles
===============

.. automethod:: openquake.vmtk.plotter.plotter.plot_demand_profiles

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.plotter import plotter

      pl = plotter()
      # peak_drift_list, peak_accel_list: one entry per record, from the
      # ``peak_drift`` / ``peak_accel`` outputs of ``do_nrha_analysis``
      pl.plot_demand_profiles(
          peak_drift_list,
          peak_accel_list,
          control_nodes,
          pFlag=True,
          export_path="demand_profiles.png",
      )
