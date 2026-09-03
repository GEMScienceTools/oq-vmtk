Nonlinear Time-History Analysis
===============================

.. automethod:: openquake.vmtk.modeller.modeller.do_nrha_analysis

.. admonition:: Example
   :class: note

   .. code-block:: python

      m.compile_model()
      m.do_gravity_analysis()
      m.do_modal_analysis(num_modes=2)
      nrha_results = m.do_nrha_analysis(
          fnames=["openquake/vmtk/tests/test_data/acceleration.txt"],
          dt_gm=0.005,
          sf=9.81,
          t_max=30.0,
          dt_ansys=0.001,
          xi=0.05,
      )
      conv_index = nrha_results[1]  # 0 = success, -1 = collapse
