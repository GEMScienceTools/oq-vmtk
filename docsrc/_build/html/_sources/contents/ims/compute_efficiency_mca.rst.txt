Efficiency — Modified Cloud Analysis
====================================

.. automethod:: openquake.vmtk.imselection.imselection.compute_efficiency_mca

.. admonition:: Theoretical Background


   Efficiency measures the dispersion of structural demand conditioned on the
   intensity measure (Luco & Cornell, 2007). A more efficient IM produces tighter
   demand predictions, reducing the required number of analyses.

   **Definition**

   For MCA, efficiency is quantified by the residual standard deviation of the
   log-log cloud regression:

   .. math::

      \beta_{D|IM} = \sqrt{
        \frac{1}{N-2} \sum_{j=1}^{N}
        \bigl[\ln(\text{EDP}_j) - \ln(a) - b\,\ln(\text{IM}_j)\bigr]^2
      }

   where :math:`a` and :math:`b` are the OLS regression coefficients of
   :math:`\ln(\text{EDP})` on :math:`\ln(\text{IM})`, and :math:`N` is the number
   of non-collapse records.

   A smaller :math:`\beta_{D|IM}` indicates that IM explains more of the
   record-to-record variability in demand — i.e. the IM is more efficient.

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.imselection import imselection

      ims = imselection()
      # cloud_dict is the output of postprocessor.process_mca_results()
      result = ims.compute_efficiency_mca(cloud_dict)
      print(f"Efficiency (beta_D|IM) = {result['efficiency']:.4f}")
