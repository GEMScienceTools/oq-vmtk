Nonlinear Time-History Animation
================================

.. automethod:: openquake.vmtk.plotter.plotter.animate_nrha

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.plotter import plotter

      pl = plotter()
      # nrha_dict from modelbuilder.do_nrha_analysis()
      pl.animate_nrha(
          nrha_dict=nrha_dict,
          export_path="nrha_animation.gif",
      )
