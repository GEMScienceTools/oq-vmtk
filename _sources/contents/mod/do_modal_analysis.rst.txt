Modal Analysis
==============

.. automethod:: openquake.vmtk.modeller.modeller.do_modal_analysis

.. admonition:: Example
   :class: note

   .. code-block:: python

      m.compile_model()
      m.do_gravity_analysis()
      periods, mode_shapes = m.do_modal_analysis(num_modes=2)
      print(f"T1 = {periods[0]:.3f} s")
