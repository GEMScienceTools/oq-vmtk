MSA Results
===========

.. automethod:: openquake.vmtk.plotter.plotter.plot_msa_analysis

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.plotter import plotter

      pl = plotter()
      # imls_matrix, edps_matrix: the MSA stripe inputs
      pl.plot_msa_analysis(
          stripe_imls=imls_matrix,    # (n_records, n_stripes)
          stripe_edps=edps_matrix,    # (n_records, n_stripes)
          imt_label="Sa(T1) [g]",
          edp_label="Peak Storey Drift [-]",
          xlims=[0, 2],
          ylims=[0, 0.05],
          export_path="msa_analysis.png",
      )
