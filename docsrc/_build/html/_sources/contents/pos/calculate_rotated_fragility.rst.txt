Fragility Function Rotation
===========================

.. automethod:: openquake.vmtk.postprocessor.postprocessor.calculate_rotated_fragility

.. admonition:: Theoretical Background

   Fragility function rotation adjusts the median capacity to account for epistemic
   uncertainty, so that the rotated curve passes through a specified percentile of the
   original curve (Porter, 2017).

   **Motivation**

   When epistemic uncertainty (building-to-building variability :math:`\beta_{\text{b2b}}`
   and damage-state threshold uncertainty :math:`\beta_{\text{DS}}`) is added to the
   aleatory record-to-record dispersion :math:`\beta_{\text{r2r}}`, the total dispersion
   increases:

   .. math::

      \beta_{\text{total}} = \sqrt{\beta_{\text{r2r}}^2 + \beta_{\text{b2b}}^2 + \beta_{\text{DS}}^2}

   A naïve approach (keeping the same median :math:`\theta`) would shift the fragility
   curve upward, implying lower damage probabilities at high intensity. The rotation
   method avoids this by anchoring the wider curve at a target percentile :math:`p` of
   the original aleatory curve.

   **Rotated median**

   The adjusted median :math:`\theta'` is chosen such that the rotated curve passes
   through the :math:`p`-th percentile of the unrotated (aleatory-only) curve:

   .. math::

      \theta' = \theta \cdot \exp\!\bigl(-\Phi^{-1}(p)\cdot
        (\beta_{\text{total}} - \beta_{\text{r2r}})\bigr)

   where :math:`\Phi^{-1}(\cdot)` is the standard normal quantile function.

   The rotated fragility curve is then:

   .. math::

      P(\text{DS} \geq ds \mid \text{IM}) =
      \Phi\!\left(\frac{\ln(\text{IM}/\theta')}{\beta_{\text{total}}}\right)

   A value of :math:`p = 0.10` (10th percentile) is a common choice, rotating the
   median downward to produce a more conservative representation of expected damage.

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.postprocessor import postprocessor

      pp = postprocessor()
      intensities = np.geomspace(0.05, 3.0, 50)
      poes = pp.calculate_rotated_fragility(
          theta=0.35,
          sigma_record2record=0.45,
          sigma_build2build=0.30,
          sigma_ds=0.30,
          intensities=intensities,
          rotation_percentile=0.10,
      )
