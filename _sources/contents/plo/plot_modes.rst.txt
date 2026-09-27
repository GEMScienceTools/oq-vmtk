Modal Shapes
============

.. automethod:: openquake.vmtk.plotter.plotter.plot_modes

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.plotter import plotter

      pl = plotter()
      # T, mode_shapes from modeller.do_modal_analysis()
      pl.plot_modes(
          node_list=model.node_list,
          mode_shape_vectors=mode_shapes,
          T=T,
          export_path="mode_shapes.png",
      )
