Postprocessing
##############

The ``postprocessor`` class derives fragility and vulnerability models from
nonlinear time-history analysis (NTHA) outputs. Supported workflows include
Modified Cloud Analysis (MCA), Multiple Stripe Analysis (MSA), and Incremental
Dynamic Analysis (IDA). Fragility fitting options are lognormal (with bootstrap or classical Bayesian
MCMC estimation), probit, logit, and ordinal models. Vulnerability functions and average annual loss (AAL) are
also computed.

.. toctree::

   pos/init
   pos/calculate_lognormal_fragility
   pos/calculate_rotated_fragility
   pos/calculate_glm_fragility
   pos/calculate_ordinal_fragility
   pos/process_mca_results
   pos/process_ida_results
   pos/process_msa_results
   pos/calculate_vulnerability_function
   pos/calculate_average_annual_damage_probability
   pos/calculate_average_annual_loss

References
----------

1. Porter, K. (2017). "When Addressing Epistemic Uncertainty in a Lognormal Fragility Function,
   How Should One Adjust the Median?", *Proceedings of the 16th World Conference on Earthquake Engineering
   (16WCEE)*, Santiago, Chile.

2. Charvet, I., Ioannou, I., Rossetto, T., Suppasri, A., and Imamura, F. (2014). "Empirical fragility
   assessment of buildings affected by the 2011 Great East Japan tsunami using improved statistical models",
   *Natural Hazards*, 73, 951-973.

3. Lallemant, D., Kiremidjian, A., and Burton, H. (2015). "Statistical procedures for developing
   earthquake damage fragility curves", *Earthquake Engineering and Structural Dynamics*, 44, 1373-1389.
   doi: 10.1002/eqe.2522.

4. Baker, J.W. (2015). "Efficient Analytical Fragility Function Fitting Using Dynamic Structural Analysis",
   *Earthquake Spectra*. 31(1):579-599. doi:10.1193/021113EQS025M

5. Nguyen, M., and Lallemant, D. (2022). "Order Matters: The Benefits of Ordinal Fragility Curves for
   Damage and Loss Estimation". *Risk Analysis*, 42: 1136-1148. https://doi.org/10.1111/risa.13815

6. Silva, V. (2019). "Uncertainty and correlation in seismic vulnerability functions of building classes."
   *Earthquake Spectra*. DOI: 10.1193/013018eqs031m.

7. Jalayer, F., De Risi, R., and Manfredi, G. (2015). "Bayesian Cloud Analysis: efficient structural
   fragility assessment using linear regression." *Bulletin of Earthquake Engineering*, 13(4), 1183–1203.

8. Cornell, C.A., Jalayer, F., Hamburger, R.O., and Foutch, D.A. (2002). "Probabilistic basis for 2000
   SAC Federal Emergency Management Agency steel moment frame guidelines." *Journal of Structural
   Engineering*, 128(4), 526–533.

9. Vamvatsikos, D. and Cornell, C.A. (2002). "Incremental dynamic analysis." *Earthquake Engineering
   and Structural Dynamics*, 31(3), 491–514.

10. McGuire, R.K. (2004). *Seismic Hazard and Risk Analysis*. Earthquake Engineering Research
    Institute, MNO-10, Oakland, CA.

11. Cornell, C.A. and Krawinkler, H. (2000). "Progress and challenges in seismic performance
    assessment." *PEER Center News*, 3(2), 1–3.

12. Jalayer, F., Ebrahimian, H., Miano, A., Manfredi, G., and Sezen, H. (2017). "Analytical
    fragility assessment using unscaled ground motion records." *Earthquake Engineering and
    Structural Dynamics*, 46: 2639–2663. https://doi.org/10.1002/eqe.2922
