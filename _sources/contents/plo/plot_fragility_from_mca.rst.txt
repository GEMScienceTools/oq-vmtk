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
          xlims=(0.0, 3.0),
          ylims=(0.0, 1.0),
          title="Fragility Functions — MCA",
          export_path="fragility_mca.png",
      )
