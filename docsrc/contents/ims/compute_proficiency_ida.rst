Proficiency — Incremental Dynamic Analysis
==========================================

.. automethod:: openquake.vmtk.imselection.imselection.compute_proficiency_ida

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.imselection import imselection

      ims = imselection()
      result = ims.compute_proficiency_ida(ida_dict)
      print(f"Proficiency (zeta) = {result['proficiency']:.4f}")
