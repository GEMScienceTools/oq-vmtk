Relative Score Method — General
===============================

.. automethod:: openquake.vmtk.imselection.imselection.compute_rsm_general

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.imselection import imselection

      ims = imselection()
      # edps, im1_values, im2_values: 1-D arrays of equal length
      result = ims.compute_rsm_general(edps, im1_values, im2_values)
      print(f"RSM = {result['rsm']:.4f} bits")
