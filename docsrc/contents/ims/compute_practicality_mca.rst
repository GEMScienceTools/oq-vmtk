Practicality — Modified Cloud Analysis
======================================

.. automethod:: openquake.vmtk.imselection.imselection.compute_practicality_mca

.. admonition:: Theoretical Background

   Practicality measures how strongly the structural demand responds to the
   intensity measure (Luco and Cornell, 2007): the steeper the IM–EDP relation,
   the more a change in IM is reflected in the demand, and the more practical
   the IM.

   **Definition**

   For MCA it is the slope :math:`b` of the log-linear cloud regression:

   .. math::

      \ln(\text{EDP}) = b_0 + b\,\ln(\text{IM})

   A higher :math:`b` indicates a stronger IM–EDP correlation.

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.imselection import imselection

      ims = imselection()
      # cloud_dict is the output of postprocessor.process_mca_results()
      result = ims.compute_practicality_mca(cloud_dict)
      print(f"Practicality (slope b) = {result['b_slope']:.4f}")
