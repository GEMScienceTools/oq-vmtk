Postprocessing Module
#####################

.. module:: postprocessor
   :synopsis: A Python module for post-processing results of nonlinear time-history
   analysis, including fragility and vulnerability analysis.

The ``postprocessor`` module provides a class ``postprocessor`` that allows users
to compute fragility functions, perform cloud and multiple stripe analyses, and
calculate vulnerability functions and average annual losses. The module supports
various fragility fitting methods, including lognormal, probit, logit, and ordinal
models. It also includes functionality to handle uncertainty and variability in
the analysis.

Classes
-------

.. class:: postprocessor()

   A class for post-processing results of nonlinear time-history analysis, including fragility and vulnerability analysis.

   **Methods**:

   .. method:: calculate_lognormal_fragility(theta, sigma_record2record, sigma_build2build=0.30, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3))

      Computes the probability of exceeding a damage state using a lognormal cumulative distribution function (CDF).

      :param theta: The median seismic intensity corresponding to a damage threshold.
      :type theta: float
      :param sigma_record2record: The logarithmic standard deviation representing record-to-record variability.
      :type sigma_record2record: float
      :param sigma_build2build: The logarithmic standard deviation representing building-to-building variability. Default is 0.30.
      :type sigma_build2build: float, optional
      :param intensities: The intensity measure levels (IMLs) at which the exceedance probabilities are computed. Default is a geometric sequence from 0.05 to 10.0 with 50 points.
      :type intensities: array-like, optional
      :return: An array of exceedance probabilities corresponding to each intensity measure level.
      :rtype: numpy.ndarray


   .. method:: calculate_rotated_fragility(theta, percentile, sigma_record2record, sigma_build2build=0.30, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3))

      Calculates a rotated fragility function based on a lognormal CDF, adjusting the median intensity to align with a specified target percentile.

      :param theta: The median seismic intensity corresponding to the damage threshold.
      :type theta: float
      :param percentile: The target percentile for fragility function rotation (e.g., 0.2 for the 20th percentile).
      :type percentile: float
      :param sigma_record2record: The uncertainty associated with record-to-record variability.
      :type sigma_record2record: float
      :param sigma_build2build: The uncertainty associated with modeling variability. Default is 0.30.
      :type sigma_build2build: float, optional
      :param intensities: The intensity measure levels (IMLs) at which the exceedance probabilities are computed. Default is a geometric sequence from 0.05 to 10.0 with 50 points.
      :type intensities: array-like, optional
      :return: The new median intensity, total standard deviation, and probabilities of exceedance.
      :rtype: tuple(float, float, array-like)


   .. method:: calculate_glm_fragility(imls, edps, damage_thresholds, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3), fragility_method='logit')

      Computes non-parametric fragility functions using Generalized Linear Models (GLM) with either a Logit or Probit link function.

      :param imls: Intensity Measure Levels (IMLs) corresponding to each observation.
      :type imls: array-like
      :param edps: Engineering Demand Parameters (EDPs) representing structural response values.
      :type edps: array-like
      :param damage_thresholds: List of thresholds defining different damage states.
      :type damage_thresholds: array-like
      :param intensities: Intensity measure values at which probabilities of exceedance (PoEs) are evaluated. Default is a geometric sequence from 0.05 to 10.0 with 50 points.
      :type intensities: array-like, optional
      :param fragility_method: Specifies the GLM model to be used for fragility function fitting. Options: 'logit' (default) or 'probit'.
      :type fragility_method: str, optional
      :return: A 2D array where each column represents the probability of exceeding a specific damage state at each intensity level.
      :rtype: numpy.ndarray


   .. method:: calculate_ordinal_fragility(imls, edps, damage_thresholds, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3))

      Fits an ordinal (cumulative) probit model to estimate fragility curves for different damage states.

      :param imls: Intensity measure levels corresponding to the observed EDPs.
      :type imls: array-like
      :param edps: Engineering Demand Parameters (EDPs) representing structural responses.
      :type edps: array-like
      :param damage_thresholds: Damage state thresholds for classifying exceedance levels.
      :type damage_thresholds: array-like
      :param intensities: Intensity measure levels for which fragility curves are evaluated. Default is a geometric sequence from 0.05 to 10.0 with 50 points.
      :type intensities: array-like, optional
      :return: A 2D array of exceedance probabilities (CDF values) for each intensity level.
      :rtype: numpy.ndarray


   .. method:: do_cloud_analysis(imls, edps, damage_thresholds, lower_limit, censored_limit, sigma_build2build=0.3, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3), fragility_rotation=False, rotation_percentile=0.1, fragility_method='lognormal')

      Perform a censored cloud analysis to assess fragility functions for a set of engineering demand parameters (EDPs) and intensity measure levels (IMLs).

      :param imls: A list or array of intensity measure levels (IMLs).
      :type imls: list or array
      :param edps: A list or array of engineering demand parameters (EDPs).
      :type edps: list or array
      :param damage_thresholds: A list of damage thresholds associated with different levels of damage.
      :type damage_thresholds: list
      :param lower_limit: The minimum value of EDP below which cloud records are excluded.
      :type lower_limit: float
      :param censored_limit: The maximum value of EDP above which cloud records are excluded.
      :type censored_limit: float
      :param sigma_build2build: The building-to-building variability or modeling uncertainty. Default is 0.3.
      :type sigma_build2build: float, optional
      :param intensities: An array of intensity measure levels used to sample and evaluate the fragility functions. Default is a geometric sequence from 0.05 to 10.0 with 50 points.
      :type intensities: array, optional
      :param fragility_rotation: A boolean flag to indicate whether or not the fragility function should be rotated about a target percentile. Default is False.
      :type fragility_rotation: bool, optional
      :param rotation_percentile: The target percentile (between 0 and 1) around which the fragility function will be rotated. Default is 0.1.
      :type rotation_percentile: float, optional
      :param fragility_method: The method used to fit the fragility function. Options: 'probit', 'logit', 'ordinal', or 'lognormal' (default).
      :type fragility_method: str, optional
      :return: A dictionary containing the outputs of the cloud analysis, including fragility functions and regression coefficients.
      :rtype: dict

   .. method:: do_multiple_stripe_analysis(imls, edps, damage_thresholds, sigma_build2build=0.3, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3), fragility_rotation=False, rotation_percentile=0.10)

      Perform maximum likelihood estimation (MLE) for fragility curve fitting following a multiple stripe analysis.

      :param imls: A list or array of intensity measure levels (IMLs).
      :type imls: list or array
      :param edps: A list or array of engineering demand parameters (EDPs).
      :type edps: list or array
      :param damage_thresholds: A list of EDP-based damage thresholds.
      :type damage_thresholds: list
      :param sigma_build2build: The building-to-building variability or modeling uncertainty. Default is 0.3.
      :type sigma_build2build: float, optional
      :param intensities: An array of intensity measure levels over which the fragility function will be sampled. Default is a geometric sequence from 0.05 to 10.0 with 50 points.
      :type intensities: array, optional
      :param fragility_rotation: A boolean flag to indicate whether or not to rotate the fragility curve about a given percentile. Default is False.
      :type fragility_rotation: bool, optional
      :param rotation_percentile: The target percentile (between 0 and 1) around which the fragility function will be rotated. Default is 0.10.
      :type rotation_percentile: float, optional
      :return: A dictionary containing the results of the multiple stripe analysis, including medians, dispersions, and probabilities of exceedance.
      :rtype: dict

   .. method:: calculate_sigma_loss(loss)

      Calculate the uncertainty in the loss estimates based on the method proposed in Silva (2019).

      :param loss: A list or array of expected loss ratios.
      :type loss: list or array
      :return: The uncertainty (sigma) associated with the mean loss ratio, and the parameters of a beta distribution (a and b).
      :rtype: tuple(list or array, list or array, list or array)


   .. method:: get_vulnerability_function(poes, consequence_model, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3), uncertainty=True)

      Calculate the vulnerability function given the probabilities of exceedance and a consequence model.

      :param poes: An array of probabilities of exceedance associated with the damage states considered.
      :type poes: array
      :param consequence_model: A list of damage-to-loss ratios corresponding to each damage state.
      :type consequence_model: list
      :param intensities: An array of intensity measure levels. Default is a geometric sequence from 0.05 to 10.0 with 50 points.
      :type intensities: array, optional
      :param uncertainty: A flag to indicate whether to calculate the coefficient of variation associated with Loss|IM. Default is True.
      :type uncertainty: bool, optional
      :return: A DataFrame containing the intensity measure levels (IML), expected loss ratios, and optionally, the coefficient of variation (COV) for each IML.
      :rtype: pandas.DataFrame

   .. method:: calculate_average_annual_damage_probability(fragility_array, hazard_array, return_period=1, max_return_period=5000)

      Calculate the Average Annual Damage State Probability (AADP) based on fragility and hazard curves.

      :param fragility_array: A 2D array where the first column contains intensity measure levels, and the second column contains the corresponding probabilities of exceedance.
      :type fragility_array: 2D array
      :param hazard_array: A 2D array where the first column contains intensity measure levels, and the second column contains the annual rates of exceedance.
      :type hazard_array: 2D array
      :param return_period: The return period used to scale the hazard rate. Default is 1.
      :type return_period: float, optional
      :param max_return_period: The maximum return period threshold used to filter out very low hazard rates. Default is 5000.
      :type max_return_period: float, optional
      :return: The average annual damage state probability.
      :rtype: float

   .. method:: calculate_average_annual_loss(vulnerability_array, hazard_array, return_period=1, max_return_period=5000)

      Calculate the Average Annual Loss (AAL) based on vulnerability and hazard curves.

      :param vulnerability_array: A 2D array where the first column contains intensity measure levels, and the second column contains the corresponding loss ratios.
      :type vulnerability_array: 2D array
      :param hazard_array: A 2D array where the first column contains intensity measure levels, and the second column contains the annual rates of exceedance.
      :type hazard_array: 2D array
      :param return_period: The return period used to scale the hazard rate. Default is 1.
      :type return_period: float, optional
      :param max_return_period: The maximum return period threshold used to filter out very low hazard rates. Default is 5000.
      :type max_return_period: float, optional
      :return: The average annual loss.
      :rtype: float


References
----------
1) Porter, K. (2017). "When Addressing Epistemic Uncertainty in a Lognormal Fragility Function,
How Should One Adjust the Median?", *Proceedings of the 16th World Conference on Earthquake Engineering
(16WCEE)*, Santiago, Chile.

2) Charvet, I., Ioannou, I., Rossetto, T., Suppasri, A., and Imamura, F. (2014). "Empirical fragility
assessment of buildings affected by the 2011 Great East Japan tsunami using improved statistical models",
*Natural Hazards*, 73, 951–973, 2014. 

3) Lahcene, E., Ioannou, I., Suppasri, A., Pakoksung, K., Paulik, R., Syamsidik, S., Bouchette, F.,
and Imamura, F. (2021). "Characteristics of building fragility curves for seismic and non-seismic tsunamis:
case studies of the 2018 Sunda Strait, 2018 Sulawesi–Palu, and 2004 Indian Ocean tsunamis,
*Natural Hazards Earth System Sciences*, 21, 2313–2344, https://doi.org/10.5194/nhess-21-2313-2021.

4) Lallemant, D., Kiremidjian, A., and Burton, H. (2015). "Statistical procedures for developing
earthquake damage fragility curves", *Earthquake Engineering and Structural Dynamics, 44, 1373–1389. doi: 10.1002/eqe.2522.

5) Jalayer, F., Ebrahamian, H., Trevlopoulos, K., and Bradley, B. (2023). "Empirical tsunami fragility modelling
for hierarchical damage levels", *Natural Hazards and Earth System Sciences*, 23(2), 909–931.
https://doi.org/10.5194/nhess-23-909-2023

6) Baker, J.W. (2015). "Efficient Analytical Fragility Function Fitting Using Dynamic Structural Analysis",
*Earthquake Spectra*. 2015;31(1):579-599. doi:10.1193/021113EQS025M

7) Singhal A., Kiremidjian AS. Method for probabilistic evaluation of seismic structural damage.
Journal of Structural Engineering 1996; 122: 1459–1467. DOI:10.1061/(ASCE)0733-9445(1996)122:12(1459)

8) Bird J.F., Bommer J.J., Bray J.D., Sancio R., Spence R.J.S., (2004). "Comparing loss estimation with observed damage in a zone
of ground failure: a study of the 1999 Kocaeli Earthquake in Turkey", *Bulletin of Earthquake Engineering* 2004; 2:
329–360. DOI: 10.1007/s10518-004-3804-0

9) Nguyen, M., and Lallemant, D. (2022). "Order Matters: The Benefits of Ordinal Fragility Curves for Damage and Loss Estimation". *Risk Analysis*,
42: 1136-1148. https://doi.org/10.1111/risa.13815

10) Silva, V. (2019). "Uncertainty and correlation in seismic vulnerability functions of building classes."
*Earthquake Spectra*. DOI: 10.1193/013018eqs031m.
