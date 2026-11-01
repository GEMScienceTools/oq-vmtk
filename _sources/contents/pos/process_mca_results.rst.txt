Modified Cloud Analysis Postprocessing
======================================

.. automethod:: openquake.vmtk.postprocessor.postprocessor.process_mca_results

.. admonition:: Theoretical Background


   The Modified Cloud Analysis (MCA) extends the classical cloud analysis approach
   (Jalayer et al., 2015) by explicitly accounting for structural collapse through a
   dual-regression procedure.

   **Step 1 — Log-log regression (non-collapse records)**

   For records that do not cause collapse (EDP < *censored_limit*), the engineering
   demand parameter is related to the intensity measure through a linear model in
   log-log space:

   .. math::

      \ln(\text{EDP}) = \ln(a) + b \cdot \ln(\text{IM}) + \varepsilon,
      \quad \varepsilon \sim \mathcal{N}(0,\, \beta_{\text{r2r}}^2)

   where :math:`a` and :math:`b` are regression coefficients and
   :math:`\beta_{\text{r2r}}` is the record-to-record dispersion.
   The median EDP given IM is:

   .. math::

      \widehat{\text{EDP}} \mid \text{IM} = a \cdot \text{IM}^{\,b}

   **Step 2 — Logistic regression (collapse probability)**

   Records exceeding *censored_limit* are classified as collapse. The probability of
   collapse conditioned on IM is estimated via logistic regression:

   .. math::

      P(C \mid \text{IM}) = \frac{1}{1 + \exp\!\bigl[-(\alpha_0 + \alpha_1\,\ln \text{IM})\bigr]}

   **Step 3 — Lognormal conditional fragility**

   For each damage state :math:`i` with threshold :math:`\delta_i`, the conditional
   (non-collapse) fragility is modelled as a lognormal CDF:

   .. math::

      P(\text{DS} \geq ds_i \mid \text{NC}, \text{IM}) =
      \Phi\!\left(\frac{\ln(\text{IM}/\theta_i)}{\beta_{\text{total}}}\right)

   where :math:`\theta_i` is the median IM capacity and the total dispersion combines
   three sources of uncertainty:

   .. math::

      \beta_{\text{total}} = \sqrt{\beta_{\text{r2r}}^2 + \beta_{\text{b2b}}^2 + \beta_{\text{DS}}^2}

   with :math:`\beta_{\text{r2r}}` the record-to-record variability,
   :math:`\beta_{\text{b2b}}` the building-to-building variability, and
   :math:`\beta_{\text{DS}}` the uncertainty in the damage-state threshold.

   **Step 4 — Total fragility**

   The total probability of exceeding each damage state is obtained by combining the
   non-collapse fragility with the collapse probability via the total probability theorem:

   .. math::

      P(\text{DS} \geq ds_i \mid \text{IM}) =
      P(\text{DS} \geq ds_i \mid \text{NC}, \text{IM}) \cdot P(\text{NC} \mid \text{IM})
      + P(C \mid \text{IM})

   where :math:`P(\text{NC} \mid \text{IM}) = 1 - P(C \mid \text{IM})`.

.. admonition:: Bootstrap vs. Classical (Bayesian MCMC) Estimation

   The ``cloud_method`` parameter controls how the 5-parameter model
   :math:`\boldsymbol{\chi} = [\ln a,\, b,\, \beta_{\text{r2r}},\, \alpha_0,\, \alpha_1]`
   is estimated.

   **Bootstrap** (``cloud_method='bootstrap'``, default): Ordinary Least Squares
   regression is repeated ``n_bootstrap`` times on resampled datasets. The fragility
   curves and their uncertainty band are derived from the percentile spread of bootstrap
   realisations.

   **Classical / Bayesian MCMC** (``cloud_method='classical'``): The joint posterior of
   :math:`\boldsymbol{\chi}` is sampled with a Metropolis–Hastings MCMC chain
   (Jalayer et al., 2017). The *robust* fragility (Eq. 10 in the reference) averages
   the lognormal CDF over the posterior, and the :math:`\pm k \cdot \sigma` confidence
   band (Eq. 11) bounds the predictive spread. The ``n_mcmc``, ``n_mcmc_burnin``, and
   ``confidence_k`` parameters control the chain length and band width.

   **Reference**: Jalayer F, Ebrahimian H, Miano A, Manfredi G, Sezen H. (2017).
   "Analytical fragility assessment using unscaled ground motion records."
   *Earthquake Engineering and Structural Dynamics*, 46: 2639–2663.
   https://doi.org/10.1002/eqe.2922

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.postprocessor import postprocessor

      pp = postprocessor()
      # imls, edps: 1-D arrays of IM levels and EDP responses from NTHA
      cloud_dict = pp.process_mca_results(
          imls=imls,
          edps=edps,
          damage_thresholds=[0.005, 0.015, 0.040],
          lower_limit=0.001,
          censored_limit=0.10,
      )
      print(cloud_dict['fragility']['medians'])

      # Ordinal CLM — constant dispersion (fully-ordered special case)
      cloud_ord_const = pp.process_mca_results(
          imls=imls,
          edps=edps,
          damage_thresholds=[0.005, 0.015, 0.040],
          lower_limit=0.001,
          censored_limit=0.10,
          fragility_method='ordinal',
          dispersion_type='constant',   # default
      )

      # Ordinal CLM — variable dispersion (general hierarchical case)
      cloud_ord_var = pp.process_mca_results(
          imls=imls,
          edps=edps,
          damage_thresholds=[0.005, 0.015, 0.040],
          lower_limit=0.001,
          censored_limit=0.10,
          fragility_method='ordinal',
          dispersion_type='variable',   # per-DS dispersions
      )

