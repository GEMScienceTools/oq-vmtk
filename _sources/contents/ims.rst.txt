Intensity Measure Selection
###########################

The ``imselection`` class evaluates and ranks intensity measures (IMs) using the
information-theoretic framework of Ebrahimian & Jalayer (2021). The central metric
is the **Relative Sufficiency Measure (RSM)**, expressed in bits: a positive
RSM(IM₂ vs IM₁) means IM₂ is the more sufficient IM. Two complementary metrics
are also provided — efficiency (βD|IM) and proficiency (βIM|DCR=1). Both Modified
Cloud Analysis (MCA) and Incremental Dynamic Analysis (IDA) workflows are supported.

.. toctree::

   ims/init
   ims/compute_efficiency_mca
   ims/compute_efficiency_ida
   ims/compute_proficiency_mca
   ims/compute_proficiency_ida
   ims/compute_rsm_mca
   ims/compute_rsm_ida
   ims/compute_rsm_general
   ims/compare_ims

References
----------

1. Ebrahimian, H. and Jalayer, F. (2021). "Selection of seismic intensity measures for
   prescribed limit states using alternative nonlinear dynamic analysis methods",
   *Earthquake Engineering and Structural Dynamics*, 50(5), 1235-1250.
   https://doi.org/10.1002/eqe.3393

2. Luco, N. and Cornell, C.A. (2007). "Structure-specific scalar intensity measures for
   near-source and ordinary earthquake ground motions." *Earthquake Spectra*, 23(2), 357–392.

3. Shome, N. and Cornell, C.A. (1999). "Probabilistic seismic demand analysis of nonlinear
   structures." Report No. RMS-35, Stanford University, Stanford, CA.

4. Padgett, J.E., Nielson, B.G., and DesRoches, R. (2008). "Selection of optimal intensity
   measures in probabilistic seismic demand models of highway bridge portfolios."
   *Earthquake Engineering & Structural Dynamics*, 37(5), 711–725.
