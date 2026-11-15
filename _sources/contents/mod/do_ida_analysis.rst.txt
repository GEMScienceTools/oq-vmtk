Incremental Dynamic Analysis
============================

.. automethod:: openquake.vmtk.modeller.modeller.do_ida_analysis

.. admonition:: Theoretical Background

   Incremental dynamic analysis (IDA) repeats the nonlinear time-history
   analysis of one ground-motion record at increasing scale factors (SF) to
   trace the IM–EDP curve up to collapse (Vamvatsikos and Cornell, 2002). To
   limit the number of runs, the method uses the *Hunt, Trace and Fill*
   algorithm (Vamvatsikos and Cornell, 2004):

   1. **Hunt**: starting from ``initial_sf``, the SF is multiplied by
      ``hunt_step`` until the peak drift reaches ``target_drift`` or the run
      collapses.
   2. **Trace**: the SF is then reduced in smaller steps, back towards the last
      non-collapsing level, to locate the collapse capacity.
   3. **Fill**: gaps between successful runs larger than ``max_fill_gap`` are
      bisected until the curve is resolved or ``max_runs`` is reached.

   Runs that fail to converge or collapse are assigned ``capping_drift`` so the
   IDA curve flatlines for visualisation. The method returns one result
   dictionary per scale factor (keys such as ``'peak_drift'``,
   ``'peak_accel'`` and ``'conv_index'``) and the list of scale factors in the
   order they were run; these are the inputs to
   ``postprocessor.process_ida_results``.

.. admonition:: Example
   :class: note

   .. code-block:: python

      m.compile_model()
      m.do_gravity_analysis()
      record_ida_results, ordered_sfs = m.do_ida_analysis(
          fnames=["openquake/vmtk/tests/test_data/acceleration.txt"],
          dt_gm=0.005,
          t_max=30.0,
          dt_ansys=0.005,
          target_drift=0.05,
          initial_sf=0.1,
          hunt_step=2.0,
          max_fill_gap=0.2,
          max_runs=15,
          xi=0.05,
      )
      for sf in sorted(record_ida_results):
          print(sf, record_ida_results[sf]["conv_index"])

References
----------

1. Vamvatsikos, D. and Cornell, C.A. (2002). "Incremental dynamic analysis",
   *Earthquake Engineering & Structural Dynamics*, 31(3), 491–514.
   https://doi.org/10.1002/eqe.141

2. Vamvatsikos, D. and Cornell, C.A. (2004). "Applied incremental dynamic
   analysis", *Earthquake Spectra*, 20(2), 523–553.
   https://doi.org/10.1193/1.1737737
