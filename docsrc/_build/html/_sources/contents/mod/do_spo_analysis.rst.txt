Static Pushover Analysis
========================

.. automethod:: openquake.vmtk.modeller.modeller.do_spo_analysis

.. admonition:: Example
   :class: note

   .. code-block:: python

      m.compile_model()
      m.do_gravity_analysis()
      periods, mode_shapes = m.do_modal_analysis(num_modes=2)
      spo_results = m.do_spo_analysis(
          push_dir=1,
          target_disp=0.10,
          phi=mode_shapes[:, 0],
          num_steps=200,
      )
