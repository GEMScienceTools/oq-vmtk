Storey Loss Functions
=====================

.. automethod:: openquake.vmtk.plotter.plotter.plot_slf_model

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.plotter import plotter

      pl = plotter()
      # out, cache from slfgenerator.generate()
      pl.plot_slf_model(
          out=out,
          cache=cache,
          edp_label="Interstorey Drift Ratio [-]",
          loss_label="Storey Loss Ratio [-]",
          xlims=[0, 0.05],
          ylims=[0, 1],
          export_path="slf_model.png",
      )
