SDOF-to-MDOF Calibration
========================

.. automethod:: openquake.vmtk.calibration.calibration.calibrate_model

.. admonition:: Theoretical Background

   The SDOF-to-MDOF calibration maps a single-degree-of-freedom (SDOF) spectral
   capacity curve to storey-level force-deformation relationships for a stick-and-mass
   MDOF model (Xu et al., 2016; Lu et al., 2020).

   **First-mode participation**

   The SDOF spectral displacement :math:`S_d` is related to the roof displacement
   :math:`u_{\text{roof}}` through the first-mode participation factor
   :math:`\Gamma_1` and the normalised first-mode shape :math:`\phi_1`:

   .. math::

      u_{\text{roof}} = \Gamma_1 \cdot \phi_1^{(n)} \cdot S_d

   where the participation factor is:

   .. math::

      \Gamma_1 = \frac{\boldsymbol{\phi}_1^T \mathbf{M} \mathbf{1}}
                      {\boldsymbol{\phi}_1^T \mathbf{M} \boldsymbol{\phi}_1}

   **Storey force distribution**

   The storey shear forces are distributed proportionally to the first-mode shape.
   For storey :math:`i`, the lateral force is:

   .. math::

      F_i = \frac{m_i \phi_1^{(i)}}{\sum_{j=1}^{n} m_j \phi_1^{(j)}} \cdot V_{\text{base}}

   where :math:`V_{\text{base}} = S_a \cdot M_{\text{eff}}` is the base shear and
   :math:`M_{\text{eff}} = \Gamma_1 \sum_i m_i \phi_1^{(i)}` is the effective modal mass.

   **Mass matrix**

   A lumped-mass idealisation assigns one translational degree of freedom per storey.
   The consistent mass matrix is diagonal:

   .. math::

      \mathbf{M} = \text{diag}(m_1, m_2, \ldots, m_n)

   with :math:`m_i = m_{\text{floor}}` for intermediate storeys and
   :math:`m_n = r_{\text{roof}} \cdot m_{\text{floor}}` at the roof (default
   :math:`r_{\text{roof}} = 0.75`).

   **Stiffness grouping and decay**

   Adjacent storeys are paired into groups of size 2. Within each group the spring
   stiffness decays with normalised height :math:`h \in [0, 1]`:

   .. math::

      k_i = k_0 \cdot \max(1.0 - 0.30\,h_i,\; 0.50)

   For soft-storey systems a further reduction factor is applied to the ground-floor
   spring to reproduce the characteristic weak-storey behaviour.

   **Period matching**

   The assembled stiffness and mass matrices are used in an eigenvalue problem
   :math:`(\mathbf{K} - \omega^2 \mathbf{M})\boldsymbol{\phi} = \mathbf{0}` to obtain
   the computed fundamental period :math:`T_{\text{computed}}`. The stiffness matrix
   is then scaled so that :math:`T_{\text{computed}} = T_{\text{target}}`, where
   :math:`T_{\text{target}}` is inferred from the first point of the SDOF capacity curve.

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.calibration import calibration

      sdof_capacity = np.array([
          [0.000, 0.00],
          [0.020, 0.18],
          [0.080, 0.22],
          [0.150, 0.10],
      ])
      cal = calibration(
          nst=4,
          sdof_capacity=sdof_capacity,
          storey_heights=[3.0, 3.0, 3.0, 3.0],
          roof_mass_factor=0.75,
      )
      storey_disps, storey_forces, masses, _ = cal.calibrate_model()
      print(f"Floor masses: {masses}")
