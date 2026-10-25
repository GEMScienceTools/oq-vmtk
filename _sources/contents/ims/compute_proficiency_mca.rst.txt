Proficiency — Modified Cloud Analysis
=====================================

.. automethod:: openquake.vmtk.imselection.imselection.compute_proficiency_mca

.. admonition:: Theoretical Background


   Proficiency combines efficiency with the predictability of the IM from seismic
   hazard analysis (Padgett et al., 2008). An IM may be efficient but difficult to
   predict via a ground-motion model (GMM), making it less practical for risk assessment.

   **Definition**

   Proficiency is defined as:

   .. math::

      \zeta = \beta_{D|IM} \cdot \sigma_{\ln IM}

   where :math:`\beta_{D|IM}` is the efficiency (record-to-record dispersion of
   demand given IM) and :math:`\sigma_{\ln IM}` is the total logarithmic standard
   deviation of the IM predicted by the GMM at the hazard level of interest (the
   "predictability" of the IM).

   For MCA, :math:`\beta_{D|IM}` is the residual standard deviation from the cloud
   regression evaluated at the DCR = 1 level.

   A smaller :math:`\zeta` indicates a more proficient IM — one that is both efficient
   and predictable.

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.imselection import imselection

      ims = imselection()
      result = ims.compute_proficiency_mca(cloud_dict)
      print(f"Proficiency (zeta) = {result['proficiency']:.4f}")


