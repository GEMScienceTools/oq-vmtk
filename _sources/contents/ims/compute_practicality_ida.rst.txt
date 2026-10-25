Practicality — Incremental Dynamic Analysis
===========================================

.. automethod:: openquake.vmtk.imselection.imselection.compute_practicality_ida

.. admonition:: Theoretical Background

   For IDA, practicality is estimated with the same log-linear convention as
   for MCA, applied to the median IDA curve: it is the ordinary-least-squares
   slope :math:`b` of :math:`\ln(\text{EDP})` on :math:`\ln(\text{IM})` along
   the median curve. A higher slope indicates a stronger IM–EDP relationship.
   The slope is NaN if the median curve has fewer than two positive-valued
   points.

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.imselection import imselection

      ims = imselection()
      # ida_dict is the output of postprocessor.process_ida_results()
      result = ims.compute_practicality_ida(ida_dict)
      print(f"Practicality (slope b) = {result['b_slope']:.4f}")
