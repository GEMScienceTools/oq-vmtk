Nonlinear Time-History Animation
================================

.. automethod:: openquake.vmtk.plotter.plotter.animate_nrha

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.plotter import plotter

      # Usual route: ``do_nrha_analysis`` (and ``do_nrha_analysis_sequences``)
      # call this method when ``save_animation_path`` is given
      m.do_nrha_analysis(
          fnames, dt_gm, sf, t_max, dt_ansys,
          save_animation_path="nrha_animation.gif",
          drift_thresholds=[0.0015, 0.0030, 0.0045, 0.0135],
      )

      # Direct call, from arrays recorded during an NRHA
      pl = plotter()
      pl.animate_nrha(
          control_nodes=control_nodes,   # node tags, base first
          acc=acc,                       # input motion [g], one per frame
          dts=dts,                       # time of each frame [s]
          nrha_disps=nrha_disps,         # (n_frames, n_nodes) [m]
          nrha_accels=nrha_accels,       # (n_frames, n_nodes) [m/s2]
          drift_thresholds=[0.0015, 0.0030, 0.0045, 0.0135],
          export_path="nrha_animation.gif",
      )
