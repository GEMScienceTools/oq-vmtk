Fragility Functions from MSA
============================

.. automethod:: openquake.vmtk.plotter.plotter.plot_fragility_from_msa

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.plotter import plotter

      pl = plotter()
      # msa_dict from postprocessor.process_msa_results()
      pl.plot_fragility_from_msa(
          msa_dict=msa_dict,
          imt_label="Sa(T1) [g]",
          ds_labels=["Slight", "Moderate", "Extensive", "Complete"],
          export_path="fragility_msa.png",
      )
