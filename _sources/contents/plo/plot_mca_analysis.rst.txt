MCA Results
===========

.. automethod:: openquake.vmtk.plotter.plotter.plot_mca_analysis

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.plotter import plotter

      pl = plotter()
      # cloud_dict from postprocessor.process_mca_results()
      pl.plot_mca_analysis(
          cloud_dict=cloud_dict,
          imt_label="Sa(T1) [g]",
          edp_label="Peak Storey Drift [-]",
          export_path="mca_analysis.png",
      )
