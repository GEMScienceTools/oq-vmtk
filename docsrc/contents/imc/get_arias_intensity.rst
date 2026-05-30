Arias Intensity
===============

.. automethod:: openquake.vmtk.imcalculator.imcalculator.get_arias_intensity

.. admonition:: Theoretical Background


   Arias Intensity (:math:`I_A`) measures the total energy of a ground-motion record
   per unit weight (Arias, 1970). It is proportional to the integral of the squared
   ground acceleration over the record duration.

   **Definition**

   .. math::

      I_A = \frac{\pi}{2g} \int_0^{T_d} \bigl[\ddot{u}_g(t)\bigr]^2 \, dt

   where :math:`\ddot{u}_g(t)` is the ground acceleration (m/s²), :math:`g` is the
   gravitational acceleration (9.81 m/s²), and :math:`T_d` is the total record
   duration.

   **Discrete approximation**

   For a digitised record with time step :math:`\Delta t` and :math:`N` samples:

   .. math::

      I_A \approx \frac{\pi}{2g} \sum_{n=1}^{N} \bigl[\ddot{u}_g(n\,\Delta t)\bigr]^2 \Delta t

   The result is expressed in m/s.

   **Significance**

   :math:`I_A` is related to the cumulative damage potential of a record and is
   commonly used to define the start and end of the strong-motion phase
   (e.g. the 5%–95% significant duration).

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.imcalculator import imcalculator

      acc = np.loadtxt("openquake/vmtk/tests/test_data/acceleration.txt")
      im = imcalculator(acc, dt=0.005)

      ai = im.get_arias_intensity()
      print(f"Arias Intensity = {ai:.3f} m/s")
