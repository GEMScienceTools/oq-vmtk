Orientation-Independent Spectral Acceleration (RotDxx)
======================================================

.. automethod:: openquake.vmtk.imcalculator.imcalculator.get_rotdxx

.. admonition:: Theoretical Background

   RotDxx is an orientation-independent intensity measure that summarises the
   spectral acceleration from two orthogonal horizontal ground-motion components
   by taking the *xx*-th percentile across all non-redundant rotation angles
   (Boore, 2010). It removes the arbitrary choice of sensor orientation and is
   therefore preferred for ground-motion selection and record scaling.

   **Rotated acceleration**

   For a rotation angle :math:`\theta \in [0°, 179°]`, the single-component
   acceleration is:

   .. math::

      a_\theta(t) = a_1(t)\cos\theta + a_2(t)\sin\theta

   where :math:`a_1` and :math:`a_2` are the two orthogonal horizontal
   components.

   **Linear superposition of SDOF responses**

   Because the SDOF oscillator is linear, the displacement response to the
   rotated excitation is:

   .. math::

      u(t,\theta) = \cos\theta\; u_1(t) + \sin\theta\; u_2(t)

   where :math:`u_1(t)` and :math:`u_2(t)` are the Newmark-:math:`\beta`
   displacement histories for :math:`a_1` and :math:`a_2` respectively.
   This means only two Newmark integrations are required (one per component),
   regardless of the number of rotation angles.

   **Spectral acceleration at angle** :math:`\theta`

   The pseudo-spectral acceleration at period :math:`T` for rotation angle
   :math:`\theta` is:

   .. math::

      SA(T,\theta) = \omega^2\,\max_t\bigl|u(t,\theta)\bigr|\Big/g

   where :math:`\omega = 2\pi/T`.

   **RotDxx definition**

   RotDxx is the *xx*-th percentile of :math:`SA(T, \theta)` over all 180
   rotation angles:

   .. math::

      \text{RotD}xx(T) = \text{percentile}_{xx}\bigl\{SA(T,\theta) :
      \theta \in \{0°, 1°, \ldots, 179°\}\bigr\}

   Common choices are **RotD50** (median, used as reference IM in ASCE 7-22)
   and **RotD100** (maximum, the largest possible single-component response).

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.imcalculator import imcalculator

      acc1 = np.loadtxt("openquake/vmtk/tests/test_data/acceleration.txt")
      acc2 = acc1 * 0.85   # synthetic orthogonal component
      im = imcalculator(acc1, dt=0.005)

      rotd50 = im.get_rotdxx(period=1.0, acc2=acc2, percentile=50)
      print(f"RotD50(T=1.0s) = {rotd50:.4f} g")
