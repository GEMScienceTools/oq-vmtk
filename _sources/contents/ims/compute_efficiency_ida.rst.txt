Efficiency — Incremental Dynamic Analysis
=========================================

.. automethod:: openquake.vmtk.imselection.imselection.compute_efficiency_ida

.. admonition:: Theoretical Background


   For IDA, efficiency is quantified by the dispersion of IM capacities across records
   at a given damage state (Shome & Cornell, 1999).

   **Definition**

   Given the IM capacity :math:`C_i^{(r)}` of record :math:`r` for damage state
   :math:`i`, the efficiency is the logarithmic standard deviation of those capacities:

   .. math::

      \beta_{D|IM} = \sqrt{
        \frac{1}{N-1} \sum_{r=1}^{N}
        \bigl(\ln C_i^{(r)} - \overline{\ln C_i}\bigr)^2
      }

   where :math:`\overline{\ln C_i} = \frac{1}{N}\sum_{r=1}^{N} \ln C_i^{(r)}`.

   A smaller dispersion indicates that records scaled to the same IM level produce
   similar structural capacities — i.e. the IM is a more efficient predictor of
   structural performance.

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.imselection import imselection

      ims = imselection()
      # ida_dict is the output of postprocessor.process_ida_results()
      result = ims.compute_efficiency_ida(ida_dict)
      print(f"Efficiency (beta_D|IM) = {result['efficiency']:.4f}")
