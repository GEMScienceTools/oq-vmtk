Cyclic Pushover Animation
=========================

.. automethod:: openquake.vmtk.plotter.plotter.animate_cpo

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.plotter import plotter

      pl = plotter()
      # cpo_dict from modelbuilder cyclic pushover analysis
      pl.animate_cpo(
          cpo_dict=cpo_dict,
          export_path="cpo_animation.gif",
      )
