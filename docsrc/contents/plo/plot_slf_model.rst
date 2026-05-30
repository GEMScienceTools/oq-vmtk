Storey Loss Functions
=====================

.. automethod:: openquake.vmtk.plotter.plotter.plot_slf_model

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.plotter import plotter

      pl = plotter()
      # slf, cache from slfgenerator.generate()
      pl.plot_slf_model(
          slf=slf,
          cache=cache,
          edp_label="Interstorey Drift Ratio [-]",
          loss_label="Storey Loss Ratio [-]",
          export_path="slf_model.png",
      )
