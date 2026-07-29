Average Spectral Acceleration
=============================

.. automethod:: openquake.vmtk.imcalculator.imcalculator.get_saavg

.. admonition:: Theoretical Background


   The Average Spectral Acceleration (AvgSA) is the geometric mean of spectral
   accelerations over a period range centred on the fundamental period of the
   structure (Cordova et al., 2000; Eads et al., 2015).

   **Definition**

   For a structure with fundamental period :math:`T_1`, AvgSA is computed over the
   range :math:`[0.2\,T_1,\, 1.5\,T_1]` at :math:`N` equally spaced periods:

   .. math::

      \text{AvgSA}(T_1) = \exp\!\left(
        \frac{1}{N} \sum_{j=1}^{N} \ln S_a(T_j)
      \right)

   This is equivalent to the geometric mean of the spectral ordinates
   :math:`S_a(T_1), S_a(T_2), \ldots, S_a(T_N)`.

   **Motivation**

   AvgSA captures the spectral shape over the range of periods most relevant to
   structural response, making it a more efficient and sufficient intensity measure
   than single-period Sa for structures that exhibit period elongation (e.g. during
   inelastic response).

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.imcalculator import imcalculator

      acc = np.loadtxt("openquake/vmtk/tests/test_data/acceleration.txt")
      im = imcalculator(acc, dt=0.005)

      avg_sa = im.get_saavg(period=0.6)
      print(f"AvgSa(T1=0.6s) = {avg_sa:.4f} g")


