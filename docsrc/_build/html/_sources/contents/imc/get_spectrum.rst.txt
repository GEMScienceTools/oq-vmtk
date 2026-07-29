Response Spectrum
=================

.. automethod:: openquake.vmtk.imcalculator.imcalculator.get_spectrum

.. admonition:: Theoretical Background


   The response spectrum is computed by solving the equation of motion of an
   undamped single-degree-of-freedom (SDOF) oscillator subjected to a base
   acceleration :math:`\ddot{u}_g(t)`, for a range of natural periods :math:`T`.

   **Equation of motion**

   .. math::

      m\,\ddot{u}(t) + c\,\dot{u}(t) + k\,u(t) = -m\,\ddot{u}_g(t)

   where :math:`m = 1` kg (unit mass), :math:`k = m\omega^2`,
   :math:`c = 2\xi m\omega`, :math:`\omega = 2\pi/T`, and :math:`\xi` is the
   damping ratio.

   **Newmark-beta integration**

   The equation of motion is integrated numerically using the Newmark constant
   average acceleration method (:math:`\gamma = 0.5`, :math:`\beta = 0.25`),
   which is unconditionally stable. At each time step :math:`n`:

   .. math::

      \tilde{k}\,\Delta u_n = \Delta p_n + A\,\dot{u}_{n-1} + B\,a_{n-1}

   where :math:`\tilde{k} = k + \tfrac{\gamma}{\beta\,\Delta t}c + \tfrac{m}{\beta\,\Delta t^2}`
   is the effective stiffness, and :math:`A`, :math:`B` are auxiliary coefficients
   derived from the Newmark parameters.

   **Spectral quantities**

   The peak responses over the record duration yield the spectral values at period :math:`T`:

   .. math::

      S_d(T) = \max_t |u(t)|, \qquad
      S_v(T) = \omega\,S_d(T), \qquad
      S_a(T) = \omega^2 S_d(T) / g

   where :math:`S_d`, :math:`S_v`, and :math:`S_a` are the spectral displacement,
   pseudo-velocity, and pseudo-acceleration respectively.

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.imcalculator import imcalculator

      acc = np.loadtxt("openquake/vmtk/tests/test_data/acceleration.txt")
      im = imcalculator(acc, dt=0.005)

      periods, sd, sv, sa = im.get_spectrum()
      print(f"Peak Sa = {max(sa):.3f} g at T = {periods[sa.argmax()]:.2f} s")
