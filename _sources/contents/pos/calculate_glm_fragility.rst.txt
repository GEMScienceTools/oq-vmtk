GLM Fragility Functions
=======================

.. automethod:: openquake.vmtk.postprocessor.postprocessor.calculate_glm_fragility

.. admonition:: Theoretical Background


   Generalised Linear Model (GLM) fragility fitting uses binary regression to estimate
   exceedance probabilities directly from damage observations, without assuming a
   lognormal form (Lallemant et al., 2015).

   **Binary response model**

   Let :math:`y_j \in \{0, 1\}` indicate whether record :math:`j` caused exceedance
   of the damage-state threshold. The exceedance probability is modelled as:

   .. math::

      P(\text{DS} \geq ds_i \mid \text{IM}_j) = g^{-1}(\eta_j),
      \quad \eta_j = \beta_0 + \beta_1 \ln(\text{IM}_j)

   where :math:`g^{-1}` is the inverse link function. Two link functions are available:

   - **Probit**: :math:`g^{-1}(\eta) = \Phi(\eta)` (standard normal CDF). This
     recovers the lognormal fragility model when the predictor is :math:`\ln(\text{IM})`.
   - **Logit**: :math:`g^{-1}(\eta) = 1/(1 + e^{-\eta})` (logistic sigmoid).

   **Maximum likelihood estimation**

   Parameters :math:`(\beta_0, \beta_1)` are estimated by maximising the Bernoulli
   log-likelihood:

   .. math::

      \mathcal{L}(\beta_0, \beta_1) =
      \sum_{j=1}^{N} \Bigl[
        y_j \ln p_j + (1 - y_j) \ln(1 - p_j)
      \Bigr],
      \quad p_j = g^{-1}(\beta_0 + \beta_1 \ln \text{IM}_j)

   **Dispersion**

   The record-to-record dispersion is derived from the fitted coefficients. For the
   probit link, :math:`\beta_{\text{r2r}} = 1/\beta_1`. Total dispersion is then:

   .. math::

      \beta_{\text{total}} = \sqrt{\beta_{\text{r2r}}^2 + \beta_{\text{b2b}}^2 + \beta_{\text{DS}}^2}

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.postprocessor import postprocessor

      pp = postprocessor()
      intensities = np.geomspace(0.05, 3.0, 50)
      # imls: 1-D array of IM levels; ds_flags: 1-D binary array (0/1) per record
      poes = pp.calculate_glm_fragility(
          imls=imls,
          ds_flags=ds_flags,
          intensities=intensities,
          method="logit",
      )
