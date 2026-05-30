Modal Shapes
============

.. automethod:: openquake.vmtk.plotter.plotter.plot_modes

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.plotter import plotter

      pl = plotter()
      # modal_dict from modelbuilder.do_modal_analysis()
      pl.plot_modes(
          modal_dict=modal_dict,
          num_modes=3,
      )
