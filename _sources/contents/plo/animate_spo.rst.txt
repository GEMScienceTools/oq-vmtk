Static Pushover Animation
=========================

.. automethod:: openquake.vmtk.plotter.plotter.animate_spo

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.plotter import plotter

      pl = plotter()
      # spo_dict from modelbuilder.do_spo_analysis()
      pl.animate_spo(
          spo_dict=spo_dict,
          export_path="spo_animation.gif",
      )
