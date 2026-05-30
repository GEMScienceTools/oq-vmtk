Fragility Functions from MCA
============================

.. automethod:: openquake.vmtk.plotter.plotter.plot_fragility_from_mca

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.plotter import plotter

      pl = plotter()
      # cloud_dict from postprocessor.process_mca_results()
      pl.plot_fragility_from_mca(
          cloud_dict=cloud_dict,
          imt_label="Sa(T1) [g]",
          ds_labels=["Slight", "Moderate", "Extensive", "Complete"],
          export_path="fragility_mca.png",
      )
