IDA Results
===========

.. automethod:: openquake.vmtk.plotter.plotter.plot_ida_analysis

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.plotter import plotter

      pl = plotter()
      # ida_dict from postprocessor.process_ida_results()
      pl.plot_ida_analysis(
          ida_dict=ida_dict,
          imt_label="Sa(T1) [g]",
          edp_label="Peak Storey Drift [-]",
          export_path="ida_analysis.png",
      )
