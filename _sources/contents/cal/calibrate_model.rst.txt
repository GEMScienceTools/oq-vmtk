SDOF-to-MDOF Calibration
========================

.. autofunction:: openquake.vmtk.calibration.calibrate_model

.. admonition:: Theoretical Background

   The SDOF-to-MDOF calibration maps a single-degree-of-freedom (SDOF) spectral
   capacity curve to storey-level force-deformation relationships for a stick-and-mass
   MDOF model (Xu et al., 2016; Lu et al., 2020).

   **First-mode shape**

   The normalised first-mode shape :math:`\phi_1` (roof value 1.0) is obtained in
   one of two ways:

   - **Frame buildings** (``is_frame=True``, :math:`n \le 12` storeys, not
     soft-storey): an assumed power law,

     .. math::

        \phi_1^{(i)} = \left(\frac{i}{n}\right)^{0.6}, \quad i = 1, \ldots, n

   - **Everything else** (including all soft-storey buildings, regardless of
     ``is_frame``): the first eigenvector of a uniform tri-diagonal lateral
     stiffness matrix (unit diagonal terms of 2, roof term of 1), solved against
     a diagonal mass matrix with a reduced roof mass factor of 0.75. For
     soft-storey systems (``is_sos=True``) the ground-floor stiffness term is
     softened to 1.20 instead of 2, so the soft-storey mode-shape softening is
     never bypassed by ``is_frame``.

   **Unit effective modal mass**

   Floor masses are scaled so that the assumed mode shape carries unit
   effective modal mass, consistent with the unit-mass SDOF capacity curve:

   .. math::

      m = \frac{\boldsymbol{\phi}_1^T \mathbf{I} \boldsymbol{\phi}_1}
               {\left(\boldsymbol{\phi}_1^T \mathbf{I} \mathbf{1}\right)^2}

   with the modal participation factor given by

   .. math::

      \Gamma_1 = \frac{\boldsymbol{\phi}_1^T \mathbf{I} \mathbf{1}}
                      {\boldsymbol{\phi}_1^T \mathbf{I} \boldsymbol{\phi}_1}

   **Storey force and drift distribution**

   The storey shear force at storey :math:`i` is the SDOF spectral acceleration
   distributed over the modal mass tributary to storeys :math:`i` through the
   roof:

   .. math::

      F_i = S_a \cdot \Gamma_1 \sum_{j=i}^{n} \phi_1^{(j)} m_j

   The first-storey drift follows directly from the SDOF spectral displacement,
   :math:`\delta_1 = S_d \cdot \Gamma_1 \cdot \phi_1^{(1)}`; every other storey
   drift is scaled by the mode-shape increment relative to the first storey:

   .. math::

      \delta_i = \delta_1 \cdot \frac{\phi_1^{(i)} - \phi_1^{(i-1)}}{\phi_1^{(1)}}

   No iterative period-matching or OpenSees verification step is performed —
   the calibration is a single closed-form pass driven entirely by the assumed
   mode shape.

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.calibration import calibrate_model

      sdof_capacity = np.array([
          [0.000, 0.00],
          [0.020, 0.18],
          [0.080, 0.22],
          [0.150, 0.10],
      ])
      floor_masses, storey_drifts, storey_forces, phi, metadata = calibrate_model(
          nst=4,
          sdof_capacity=sdof_capacity,
          storey_heights=[3.0, 3.0, 3.0, 3.0],
      )
      print(f"Floor masses: {floor_masses}")
