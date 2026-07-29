Average Annual Damage Probabilities
===================================

.. automethod:: openquake.vmtk.postprocessor.postprocessor.calculate_average_annual_damage_probability

.. admonition:: Theoretical Background

   The Average Annual Damage Probability (AADP) for a damage state :math:`ds` is the
   expected frequency of exceeding that damage state per year, obtained by integrating
   the fragility curve over the seismic hazard (McGuire, 2004).

   **Classical integral**

   .. math::

      \text{AADP}_{ds} =
      \int_0^{\infty}
        P(\text{DS} \geq ds \mid \text{IM} = x)\;
        \left|\frac{d\lambda(x)}{dx}\right| dx

   where:

   - :math:`P(\text{DS} \geq ds \mid \text{IM} = x)` is the fragility curve giving
     the probability of exceeding damage state :math:`ds` at intensity :math:`x`,
   - :math:`\lambda(x) = P(\text{IM} > x)` is the mean annual rate of exceedance
     from the hazard curve, and
   - :math:`|d\lambda/dx|` is the probability density of IM occurrences per year.

   **Discrete approximation**

   In practice the integral is evaluated numerically using midpoint quadrature over
   the IM bins of the hazard curve:

   .. math::

      \text{AADP}_{ds} \approx
      \sum_{j} P(\text{DS} \geq ds \mid \text{IM} = \bar{x}_j) \cdot
      \Delta\lambda_j

   where :math:`\bar{x}_j = (x_j + x_{j+1})/2` is the midpoint of the :math:`j`-th
   IM interval and :math:`\Delta\lambda_j = |\lambda(x_j) - \lambda(x_{j+1})|` is
   the corresponding rate of occurrence.

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.postprocessor import postprocessor

      pp = postprocessor()
      # fragility_medians, fragility_betas: arrays from process_mca_results()
      # hazard_imls, hazard_poes: hazard curve arrays
      aadp = pp.calculate_average_annual_damage_probability(
          fragility_medians=fragility_medians,
          fragility_betas=fragility_betas,
          hazard_imls=hazard_imls,
          hazard_poes=hazard_poes,
      )
      print(aadp)
