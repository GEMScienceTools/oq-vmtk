RotDxx of Other Intensity Measures
==================================

.. automethod:: openquake.vmtk.imcalculator.imcalculator.get_rotdxx_amplitude_ims

.. automethod:: openquake.vmtk.imcalculator.imcalculator.get_rotdxx_cav

.. automethod:: openquake.vmtk.imcalculator.imcalculator.get_rotdxx_arias_duration

.. automethod:: openquake.vmtk.imcalculator.imcalculator.get_rotdxx_saavg

.. automethod:: openquake.vmtk.imcalculator.imcalculator.get_rotdxx_FIV3

.. admonition:: Theoretical Background

   ``get_rotdxx`` gives the RotDxx *response spectrum*. The methods above
   extend the same definition (Boore, 2010) to the other intensity measures:
   the IM is evaluated on the rotated accelerogram

   .. math::

      a_\theta(t) = a_1(t)\cos\theta + a_2(t)\sin\theta,
      \qquad \theta \in \{0°, 1°, \ldots, 179°\}

   and RotDxx is the *xx*-th percentile of the 180 values (RotD50 is the
   median, RotD100 the maximum). The percentile is taken **of the IM**, not of
   an intermediate quantity, so for example the RotD50 of AvgSA is the median
   over angles of the AvgSA of each rotated record, which is not the AvgSA of
   the RotD50 spectrum.

   The cost depends on how the IM depends on the record:

   * **Linear quantities** (PGA, PGV, PGD, the SDOF displacement histories
     behind SA/AvgSA, and the FIV series of FIV3) are computed once per
     component and *rotated*, :math:`x_\theta = x_1\cos\theta + x_2\sin\theta`.
     The result is identical to recomputing at each angle. For PGV and PGD the
     drift-corrected histories of ``get_vel_disp_history`` are rotated, so the
     ``highpass_hz`` argument applies as in ``get_amplitude_ims``.
   * **Arias Intensity and significant duration** are quadratic in the
     record. The rotated cumulative energy is assembled from the cumulative
     sums of :math:`a_1^2`, :math:`a_2^2` and :math:`a_1 a_2`:

     .. math::

        E_\theta(t) = \cos^2\theta\, E_{11}(t) + \sin^2\theta\, E_{22}(t)
        + 2\sin\theta\cos\theta\, E_{12}(t)

     so no rotated record is built; the duration is the time between the
     requested fractions of :math:`E_\theta / E_\theta(T_d)`.
   * **CAV** involves :math:`|a_\theta|` and is recomputed at each angle.
   * **FIV3** rotates the two FIV series; only the peak and trough picking is
     repeated at each angle.

   **AvgSA.** The geometric mean over the periods is taken at each angle and
   the oscillators are integrated at the exact periods (no interpolation on a
   spectrum grid). Either a conditioning period (10 periods in
   :math:`[0.2T, 1.5T]`, as in ``get_saavg``) or a user-defined list of
   periods can be given.

   **Units** are those of the single-component methods: PGA in g, PGV in m/s,
   PGD in m, CAV and Arias Intensity in m/s, duration in s, AvgSA in g and
   FIV3 in g·s. Both components must have the same length and unit.

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.imcalculator import imcalculator

      acc1 = np.loadtxt("openquake/vmtk/tests/test_data/acceleration.txt")
      acc2 = acc1 * 0.85   # synthetic orthogonal component
      im = imcalculator(acc1, dt=0.005)

      pga, pgv, pgd = im.get_rotdxx_amplitude_ims(acc2, percentile=50)
      cav = im.get_rotdxx_cav(acc2)
      ai, d595 = im.get_rotdxx_arias_duration(acc2)
      avgsa = im.get_rotdxx_saavg(acc2, period=1.0)
      fiv3 = im.get_rotdxx_FIV3(acc2, period=1.0, alpha=0.7, beta=0.85)
      print(f"RotD50 PGA = {pga:.4f} g, AvgSA(1.0 s) = {avgsa:.4f} g")
