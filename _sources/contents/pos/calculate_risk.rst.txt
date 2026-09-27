Average Annual Risk Metrics
============================

.. automethod:: openquake.vmtk.postprocessor.postprocessor.calculate_risk

.. admonition:: Theoretical Background

   ``calculate_risk`` integrates an intensity-measure-dependent curve against
   a seismic hazard curve to obtain an average annual risk metric. The same
   integral applies regardless of what the curve represents — only its
   interpretation changes:

   - Passing a **fragility curve** (probability of exceeding a damage state
     vs. intensity) yields the Average Annual Damage Probability (AADP)
     for that damage state (McGuire, 2004).
   - Passing a **vulnerability curve** (expected loss ratio vs. intensity)
     yields the Average Annual Loss Ratio (AALR) (Cornell & Krawinkler,
     2000).

   **Classical integral**

   .. math::

      \text{Risk} =
      \int_0^{\infty}
        Y(\text{IM} = x)\;
        \left|\frac{d\lambda(x)}{dx}\right| dx

   where:

   - :math:`Y(\text{IM} = x)` is the value of the input curve at intensity
     :math:`x` — either :math:`P(\text{DS} \geq ds \mid \text{IM} = x)` for
     a fragility curve, or :math:`E[L \mid \text{IM} = x]` for a
     vulnerability curve,
   - :math:`\lambda(x) = P(\text{IM} > x)` is the mean annual rate of
     exceedance from the hazard curve, and
   - :math:`|d\lambda/dx|` is the probability density of IM occurrences
     per year.

   **Discrete approximation**

   In practice the integral is evaluated numerically using midpoint
   quadrature over the IM bins of the hazard curve:

   .. math::

      \text{Risk} \approx
      \sum_{j} Y(\text{IM} = \bar{x}_j) \cdot \Delta\lambda_j

   where :math:`\bar{x}_j = (x_j + x_{j+1})/2` is the midpoint of the
   :math:`j`-th IM interval and :math:`\Delta\lambda_j = |\lambda(x_j) -
   \lambda(x_{j+1})|` is the corresponding rate of occurrence.

   Intensity levels with exceedance rates below :math:`1/T_{\max}` (where
   :math:`T_{\max}` is the maximum return period) are excluded to avoid
   numerical instability from very rare events.

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.postprocessor import postprocessor

      pp = postprocessor()
      # vulnerability_array: array of [IML, mean loss ratio] pairs
      # fragility_array: array of [IML, probability of exceedance] pairs
      # hazard_array: array of [IML, annual rate of exceedance] pairs
      aalr = pp.calculate_risk(
          input_array=vulnerability_array,
          hazard_array=hazard_array,
      )
      print(f"AALR = {aalr:.4f}")

      aadp = pp.calculate_risk(
          input_array=fragility_array,
          hazard_array=hazard_array,
      )
      print(f"AADP = {aadp:.4f}")
