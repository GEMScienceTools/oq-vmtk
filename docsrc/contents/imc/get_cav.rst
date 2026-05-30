Cumulative Absolute Velocity
============================

.. automethod:: openquake.vmtk.imcalculator.imcalculator.get_cav

.. admonition:: Theoretical Background


   Cumulative Absolute Velocity (CAV) is the integral of the absolute value of ground
   acceleration over the record duration (Kempton & Stewart, 2006). It captures both
   the amplitude and duration of shaking.

   **Definition**

   .. math::

      \text{CAV} = \int_0^{T_d} \bigl|\ddot{u}_g(t)\bigr| \, dt

   where :math:`\ddot{u}_g(t)` is the ground acceleration (m/s²) and
   :math:`T_d` is the total record duration.

   **Discrete approximation**

   For a digitised record with time step :math:`\Delta t`:

   .. math::

      \text{CAV} \approx \sum_{n=1}^{N} \bigl|\ddot{u}_g(n\,\Delta t)\bigr| \Delta t

   The result is expressed in m/s.

   **Significance**

   CAV is closely related to the potential for structural damage and liquefaction.
   Unlike peak ground motion parameters, CAV accounts for the duration of shaking,
   making it a more informative measure for cumulative damage assessment.

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.imcalculator import imcalculator

      acc = np.loadtxt("openquake/vmtk/tests/test_data/acceleration.txt")
      im = imcalculator(acc, dt=0.005)

      cav = im.get_cav()
      print(f"CAV = {cav:.3f} m/s")
