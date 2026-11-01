Relative Score Method — Modified Cloud Analysis
===============================================

.. automethod:: openquake.vmtk.imselection.imselection.compute_rsm_mca

.. admonition:: Theoretical Background


   The Relative Score Method (RSM) quantifies the information gain from using one
   intensity measure over another using a relative sufficiency metric expressed in
   bits (Ebrahimian & Jalayer, 2021).

   **Kullback–Leibler divergence**

   The sufficiency of IM₂ relative to IM₁ is measured by the Kullback–Leibler (KL)
   divergence between the demand distributions conditioned on each IM.
   For two lognormal distributions :math:`D|IM_1 \sim \ln\mathcal{N}(\mu_1, \sigma_1^2)`
   and :math:`D|IM_2 \sim \ln\mathcal{N}(\mu_2, \sigma_2^2)`, the KL divergence is:

   .. math::

      D_{KL}(f_1 \| f_2) = \ln\frac{\sigma_2}{\sigma_1}
      + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}

   **Relative Sufficiency Measure**

   The RSM of IM₂ relative to IM₁, averaged over the record set, is:

   .. math::

      \text{RSM}(\text{IM}_2 \mid \text{IM}_1) =
      \frac{1}{N} \sum_{j=1}^{N}
      \ln\frac{f_{D|\text{IM}_1}(\text{EDP}_j)}
              {f_{D|\text{IM}_2}(\text{EDP}_j)}

   expressed in nats (divided by :math:`\ln 2` to convert to bits). A positive RSM
   means IM₂ is the more sufficient IM.

   For MCA, the conditional demand distributions are derived from the cloud regression
   residuals evaluated at each record's demand and IM level.

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.imselection import imselection

      ims = imselection()
      # Compare cloud_dict2 (IM2) against cloud_dict1 (IM1)
      result = ims.compute_rsm_mca(cloud_dict1, cloud_dict2)
      print(f"RSM(IM2 vs IM1) = {result['rsm']:.4f} bits")
