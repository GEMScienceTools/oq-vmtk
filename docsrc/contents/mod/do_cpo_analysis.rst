Cyclic Pushover Analysis
========================

.. automethod:: openquake.vmtk.modeller.modeller.do_cpo_analysis

.. admonition:: Theoretical Background

   The cyclic pushover (CPO) analysis applies a lateral load pattern
   :math:`\phi` (scaled by the floor masses) under displacement control at the
   control node, following a reversed-cyclic protocol instead of a single
   monotonic excursion. Each cycle level :math:`i` reaches a target
   displacement

   .. math::

      \delta_i = \mu_i \, \delta_{ref}

   where :math:`\delta_{ref}` is the reference displacement ``ref_disp`` (e.g.
   the yield displacement) and :math:`\mu_i` the ductility levels in
   ``mu_levels``. Every half-cycle is subdivided into at least ``dispIncr``
   increments; if ``max_step`` is given, the number of increments is
   :math:`\max(\texttt{dispIncr}, \lceil \text{excursion}/\texttt{max\_step}
   \rceil)` so larger cycles are integrated with the same resolution.

   The resulting hysteresis loops expose strength degradation, stiffness
   deterioration, pinching and the energy dissipation capacity of the storey
   springs, which a monotonic pushover cannot show. Collapse is detected as in
   the other analyses: through the MinMax material limit and an independent
   cross-check of the nodal inter-storey displacement.

.. admonition:: Example
   :class: note

   .. code-block:: python

      m.compile_model()
      m.do_gravity_analysis()
      periods, mode_shapes = m.do_modal_analysis(num_modes=2)
      cpo_dict = m.do_cpo_analysis(
          ref_disp=0.005,
          mu_levels=[1, 2, 4, 6],
          push_dir=1,
          dispIncr=10,
          phi=mode_shapes[:, 0],
          max_step=0.0005,
          save_animation_path="cpo_animation.gif",  # or None
      )
