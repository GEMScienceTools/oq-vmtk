MSA Results
===========

.. automethod:: openquake.vmtk.plotter.plotter.plot_msa_analysis

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.plotter import plotter

      pl = plotter()
      # msa_dict from postprocessor.process_msa_results()
      pl.plot_msa_analysis(
          msa_dict=msa_dict,
          imt_label="Sa(T1) [g]",
          edp_label="Peak Storey Drift [-]",
          export_path="msa_analysis.png",
      )
