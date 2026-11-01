Static Pushover Animation
=========================

.. automethod:: openquake.vmtk.plotter.plotter.animate_spo

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.plotter import plotter

      # Usual route: ``do_spo_analysis`` calls this method when
      # ``save_animation_path`` is given
      spo_dict = m.do_spo_analysis(
          ref_disp=0.005, disp_scale_factor=20, push_dir=1, phi=phi,
          save_animation_path="spo_animation.gif",
      )

      # Direct call, from the arrays in ``spo_dict``
      pl = plotter()
      pl.animate_spo(
          spo_top_disp=spo_dict["spo_disps"][:, -1],   # roof displacement [m]
          spo_rxn=spo_dict["spo_rxn"],
          spo_disps=spo_dict["spo_disps"],
          spo_midr=spo_dict["spo_midr"],
          nodeList=nodeList,
          elementList=elementList,
          push_dir=1,
          phi=phi,                       # floor load pattern, no base node
          export_path="spo_animation.gif",
      )
