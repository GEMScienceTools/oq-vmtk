Multiple Stripe Analysis Postprocessing
=======================================

.. automethod:: openquake.vmtk.postprocessor.postprocessor.process_msa_results

.. admonition:: Theoretical Background


   Multiple Stripe Analysis (MSA) conditions structural response on discrete intensity
   levels (stripes), fitting fragility functions directly from binomial observations at
   each stripe (Baker, 2015).

   **Observed data per stripe**

   At each intensity level :math:`\text{IM}_j`, :math:`n_j` ground-motion records are
   run and :math:`z_j` records result in exceedance of the damage-state threshold.
   The empirical exceedance rate at stripe :math:`j` is :math:`\hat{p}_j = z_j / n_j`.

   **Maximum likelihood estimation**

   Assuming a lognormal fragility model
   :math:`F(\text{IM};\,\theta,\beta) = \Phi\!\bigl(\ln(\text{IM}/\theta)/\beta\bigr)`,
   the parameters :math:`(\theta, \beta)` are estimated by maximising the binomial
   log-likelihood across all stripes:

   .. math::

      \mathcal{L}(\theta, \beta) =
      \sum_{j=1}^{m} \left[
        z_j \ln F(\text{IM}_j;\,\theta,\beta)
        + (n_j - z_j) \ln\bigl(1 - F(\text{IM}_j;\,\theta,\beta)\bigr)
      \right]

   where :math:`m` is the total number of stripes, :math:`\theta` is the median IM
   capacity, and :math:`\beta` is the logarithmic standard deviation.

   **Total dispersion**

   The fitted :math:`\beta_{\text{r2r}}` represents record-to-record variability.
   Additional modelling uncertainties are combined in quadrature:

   .. math::

      \beta_{\text{total}} = \sqrt{\beta_{\text{r2r}}^2 + \beta_{\text{b2b}}^2 + \beta_{\text{DS}}^2}

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.postprocessor import postprocessor

      pp = postprocessor()
      # stripe_imls, stripe_edps: 2-D arrays (n_records × n_stripes) from MSA
      msa_dict = pp.process_msa_results(
          stripe_imls=stripe_imls,
          stripe_edps=stripe_edps,
          damage_thresholds=[0.005, 0.015, 0.040],
      )
      print(msa_dict['fragility']['medians'])
