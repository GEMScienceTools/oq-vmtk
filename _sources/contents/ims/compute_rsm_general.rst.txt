Relative Score Method — General
===============================

.. automethod:: openquake.vmtk.imselection.imselection.compute_rsm_general

.. admonition:: Example
   :class: note

   .. code-block:: python

      from openquake.vmtk.imselection import imselection

      import numpy as np
      from scipy.stats import lognorm

      ims = imselection()

      # f(d, im): density of the demand d given the IM, here a lognormal
      # model ln(d) ~ N(ln(a) + b ln(im), sigma) with user-fitted (a, b, sigma)
      def pdf_im1(d, im, a=0.01, b=1.0, sigma=0.4):
          return lognorm.pdf(d, s=sigma, scale=a * im**b)

      def pdf_im2(d, im, a=0.012, b=0.9, sigma=0.6):
          return lognorm.pdf(d, s=sigma, scale=a * im**b)

      # demands, im1_values, im2_values: 1-D arrays of equal length
      result = ims.compute_rsm_general(
          demands, im1_values, im2_values, pdf_im1, pdf_im2
      )
      print(f"RSM = {result['rsm']:.4f} bits")
