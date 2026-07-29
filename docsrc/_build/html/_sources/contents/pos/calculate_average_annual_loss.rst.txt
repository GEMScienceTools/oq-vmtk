Average Annual Loss
===================

.. automethod:: openquake.vmtk.postprocessor.postprocessor.calculate_average_annual_loss

.. admonition:: Theoretical Background

   The Average Annual Loss Ratio (AALR) is the expected loss per year normalised by
   the replacement cost, obtained by integrating the vulnerability function over the
   seismic hazard (Cornell & Krawinkler, 2000).

   **Classical integral**

   .. math::

      \text{AALR} =
      \int_0^{\infty}
        E[L \mid \text{IM} = x]\;
        \left|\frac{d\lambda(x)}{dx}\right| dx

   where:

   - :math:`E[L \mid \text{IM} = x]` is the expected loss ratio at intensity :math:`x`
     (the vulnerability function),
   - :math:`\lambda(x) = P(\text{IM} > x)` is the mean annual rate of exceedance
     from the hazard curve, and
   - :math:`|d\lambda/dx|` is the probability density of IM occurrences per year.

   **Discrete approximation**

   In practice the integral is evaluated numerically using midpoint quadrature:

   .. math::

      \text{AALR} \approx
      \sum_{j} E[L \mid \text{IM} = \bar{x}_j] \cdot \Delta\lambda_j

   where :math:`\bar{x}_j = (x_j + x_{j+1})/2` is the midpoint of the :math:`j`-th
   IM interval and :math:`\Delta\lambda_j = |\lambda(x_j) - \lambda(x_{j+1})|` is
   the corresponding rate of occurrence.

   Intensity levels with exceedance rates below :math:`1/T_{\max}` (where
   :math:`T_{\max}` is the maximum return period) are excluded to avoid numerical
   instability from very rare events.

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.postprocessor import postprocessor

      pp = postprocessor()
      # vulnerability: array of mean loss ratios at each IM level
      # hazard_imls, hazard_poes: hazard curve arrays
      aal = pp.calculate_average_annual_loss(
          vulnerability=vulnerability,
          hazard_imls=hazard_imls,
          hazard_poes=hazard_poes,
      )
      print(f"AAL = {aal:.4f}")
