Fragility Functions from IDA
============================

.. automethod:: openquake.vmtk.plotter.plotter.plot_fragility_from_ida

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.plotter import plotter

      pl = plotter()
      # ida_dict from postprocessor.process_ida_results()
      pl.plot_fragility_from_ida(
          ida_dict=ida_dict,
          imt_label="Sa(T1) [g]",
          ds_labels=["Slight", "Moderate", "Extensive", "Complete"],
          export_path="fragility_ida.png",
      )
