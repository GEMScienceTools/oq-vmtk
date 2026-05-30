Filtered Incremental Velocity
=============================

.. automethod:: openquake.vmtk.imcalculator.imcalculator.get_FIV3

.. admonition:: Theoretical Background


   The Filtered Incremental Velocity (FIV3) is a duration- and pulse-sensitive
   intensity measure designed to improve collapse prediction for structures with
   period elongation (Dávalos & Miranda, 2019).

   **Velocity time history**

   The ground velocity is obtained by integrating the acceleration record:

   .. math::

      v(t) = \int_0^t \ddot{u}_g(\tau)\, d\tau

   **Band-pass filtering**

   The velocity is band-pass filtered to retain energy in the period range
   :math:`[\alpha T,\, \beta T]`, where :math:`T` is the structural period of
   interest and :math:`\alpha`, :math:`\beta` are user-defined scale factors
   (typically :math:`\alpha = 1.0`, :math:`\beta = 0.7` relative to :math:`T`).
   The filter isolates the frequency content most damaging to the structure.

   **Incremental velocity pulses**

   The filtered velocity :math:`v_f(t)` is scanned for pulses — intervals between
   consecutive zero-crossings. The incremental velocity of each pulse is its
   peak-to-trough amplitude:

   .. math::

      \Delta v_k = \max_{t \in \text{pulse}_k} v_f(t) - \min_{t \in \text{pulse}_k} v_f(t)

   **FIV3**

   FIV3 is the sum of the three largest incremental velocity pulses:

   .. math::

      \text{FIV3} = \Delta v_{(1)} + \Delta v_{(2)} + \Delta v_{(3)}

   where :math:`\Delta v_{(1)} \geq \Delta v_{(2)} \geq \Delta v_{(3)}` are the
   three largest values in descending order.

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.imcalculator import imcalculator

      acc = np.loadtxt("openquake/vmtk/tests/test_data/acceleration.txt")
      im = imcalculator(acc, dt=0.005)

      fiv3 = im.get_FIV3(period=0.3)
      print(f"FIV3(T=0.3s) = {fiv3:.4f} g")
