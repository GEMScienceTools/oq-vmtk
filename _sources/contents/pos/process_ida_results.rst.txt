Incremental Dynamic Analysis Postprocessing
===========================================

.. automethod:: openquake.vmtk.postprocessor.postprocessor.process_ida_results

.. admonition:: Theoretical Background


   Incremental Dynamic Analysis (IDA) scales each record to multiple intensity levels
   and traces the structural response from elastic behaviour to collapse, producing
   IDA curves and a statistical basis for fragility derivation (Vamvatsikos & Cornell, 2002).

   **IDA curves**

   For each ground-motion record :math:`r` and scaling factor :math:`\lambda`, the
   scaled IM is :math:`\text{IM}_{r,\lambda} = \lambda \cdot \text{IM}_{r,1}` and the
   resulting peak EDP is recorded. The full set of :math:`(\text{IM}, \text{EDP})` pairs
   for record :math:`r` forms its IDA curve.

   **Limit-state capacity**

   The IM capacity :math:`C_i^{(r)}` of record :math:`r` for damage state :math:`i`
   is the smallest IM at which the IDA curve first exceeds the threshold
   :math:`\delta_i`:

   .. math::

      C_i^{(r)} = \inf\bigl\{\text{IM} : \text{EDP}(\text{IM}) \geq \delta_i\bigr\}

   **Lognormal fragility fitting**

   Across all records, the capacities :math:`\{C_i^{(r)}\}` are assumed lognormally
   distributed. The median :math:`\theta_i` and logarithmic standard deviation
   :math:`\beta_{\text{rr}}` are estimated by maximum likelihood:

   .. math::

      \theta_i = \exp\!\left(\frac{1}{N}\sum_{r=1}^{N} \ln C_i^{(r)}\right), \qquad
      \beta_{\text{r2r}} = \sqrt{\frac{1}{N-1}\sum_{r=1}^{N}\bigl(\ln C_i^{(r)} - \ln \theta_i\bigr)^2}

   **Total dispersion**

   Additional sources of uncertainty are combined in quadrature:

   .. math::

      \beta_{\text{total}} = \sqrt{\beta_{\text{r2r}}^2 + \beta_{\text{b2b}}^2 + \beta_{\text{DS}}^2}

   **Fragility function**

   The probability of exceeding damage state :math:`i` at a given IM is:

   .. math::

      P(\text{DS} \geq ds_i \mid \text{IM}) =
      \Phi\!\left(\frac{\ln(\text{IM}/\theta_i)}{\beta_{\text{total}}}\right)

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.postprocessor import postprocessor

      pp = postprocessor()
      # stripe_imls, stripe_edps: 2-D arrays (n_records × n_stripes) from IDA
      ida_dict = pp.process_ida_results(
          stripe_imls=stripe_imls,
          stripe_edps=stripe_edps,
          damage_thresholds=[0.005, 0.015, 0.040],
          imt_key="Sa(T1)",
      )
      print(ida_dict['fragility']['medians'])
