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
          ref_disp=0.005,
          disp_scale_factor=20,
          push_dir=1,
          phi=mode_shapes[:, 0],
          num_steps=200,
      )
      print(f"conv_index = {spo_results['conv_index']}")  # 0 = success, -1 = collapse
