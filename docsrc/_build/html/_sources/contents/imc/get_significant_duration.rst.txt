Significant Duration
====================

.. automethod:: openquake.vmtk.imcalculator.imcalculator.get_significant_duration

.. admonition:: Theoretical Background


   Significant duration quantifies the time interval over which the most energetic
   portion of a ground-motion record occurs, defined by the accumulation of Arias
   Intensity between two threshold fractions (Trifunac & Brady, 1975).

   **Arias Intensity accumulation**

   The normalised cumulative Arias Intensity at time :math:`t` is:

   .. math::

      I_A^*(t) = \frac{1}{I_A} \cdot \frac{\pi}{2g}
      \int_0^{t} \bigl[\ddot{u}_g(\tau)\bigr]^2 d\tau

   where :math:`I_A` is the total Arias Intensity of the record.

   **Significant duration**

   The significant duration between fractions :math:`p_1` and :math:`p_2` is:

   .. math::

      D_{p_1\text{–}p_2} = t(I_A^* = p_2) - t(I_A^* = p_1)

   The default interval is 5%–95% (:math:`D_{5\text{–}95}`), which captures the
   time between the onset and end of strong shaking while excluding the low-energy
   tails of the record.

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.imcalculator import imcalculator

      acc = np.loadtxt("openquake/vmtk/tests/test_data/acceleration.txt")
      im = imcalculator(acc, dt=0.005)

      t595 = im.get_significant_duration()
      print(f"D5-95 = {t595:.3f} s")


