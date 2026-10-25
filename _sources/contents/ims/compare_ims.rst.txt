IM Comparison
=============

.. automethod:: openquake.vmtk.imselection.imselection.compare_ims

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.imselection import imselection

      ims = imselection()
      # cloud_pga, cloud_sa, cloud_avgsa: outputs of postprocessor.process_mca_results()
      results = {
          "PGA":    cloud_pga,
          "Sa(T1)": cloud_sa,
          "AvgSa":  cloud_avgsa,
      }
      ranking = ims.compare_ims(results, analysis_type="MCA", metric="all")
      print(ranking["ranking"])
