Relative Score Method — Incremental Dynamic Analysis
====================================================

.. automethod:: openquake.vmtk.imselection.imselection.compute_rsm_ida

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.imselection import imselection

      ims = imselection()
      result = ims.compute_rsm_ida(ida_dict1, ida_dict2)
      print(f"RSM(IM2 vs IM1) = {result['rsm']:.4f} bits")
