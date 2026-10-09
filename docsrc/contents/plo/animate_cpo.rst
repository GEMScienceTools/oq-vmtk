Cyclic Pushover Animation
=========================

.. automethod:: openquake.vmtk.plotter.plotter.animate_cpo

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.plotter import plotter

      # Usual route: ``do_cpo_analysis`` calls this method when
      # ``save_animation_path`` is given
      cpo_dict = m.do_cpo_analysis(
          ref_disp, mu_levels, push_dir, dispIncr, phi,
          save_animation_path="cpo_animation.gif",
      )

      # Direct call, from the dictionary returned by ``do_cpo_analysis``
      pl = plotter()
      pl.animate_cpo(
          cpo_dict=cpo_dict,
          nodeList=nodeList,
          elementList=elementList,
          push_dir=push_dir,
          export_path="cpo_animation.gif",
      )
