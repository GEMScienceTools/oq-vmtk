Ordinal Fragility Functions
===========================

.. automethod:: openquake.vmtk.postprocessor.postprocessor.calculate_ordinal_fragility

.. admonition:: Theoretical Background


   **Hierarchical fragility model**

   The hierarchical (cumulative link) approach treats the probability of
   equalling or exceeding damage state :math:`i` given intensity measure
   level IM as a lognormal CDF:

   .. math::

      P(\mathrm{DS} \geq ds_i \mid \mathrm{IM}) =
      \Phi\!\left(\frac{\ln(\mathrm{IM}) - \ln(\theta_i)}{\beta_i}\right),
      \quad i = 1, \ldots, k

   where :math:`\theta_i` is the median IM capacity for damage state
   :math:`i` and :math:`\beta_i` is its associated dispersion. All damage
   states are fitted simultaneously (or with an ordering constraint
   applied post-fit), exploiting the ordinal structure of the damage
   scale. The **hierarchical constraint**

   .. math::

      \theta_1 \leq \theta_2 \leq \cdots \leq \theta_k

   ensures that the 50th-percentile capacities are ordered, i.e. higher
   damage states are first exceeded at larger intensity levels
   (Jalayer et al., 2023; Nguyen & Lallemant, 2022).

   **Constant dispersion** (``dispersion_type='constant'``, default)

   When a single dispersion :math:`\beta_i = \beta` is assumed for all
   damage states — the *fully ordered* special case — the model reduces
   to the standard ordered-probit parameterisation with :math:`k` ordered
   intercepts :math:`\alpha_i` and one common slope:

   .. math::

      P(\mathrm{DS} \geq ds_i \mid \mathrm{IM}) =
      \Phi(\alpha_i - \beta\,\ln \mathrm{IM}),
      \quad \alpha_1 \leq \cdots \leq \alpha_k

   The shared slope is estimated by fitting a single
   ``statsmodels.OrderedModel`` (probit distribution) to all
   damage-state observations simultaneously. Because every curve has
   identical shape and only the threshold differs, crossings are
   strictly impossible **everywhere** along the IM axis.

   **Variable dispersion** (``dispersion_type='variable'``)

   In general, each damage state may exhibit its own sensitivity to
   ground-motion intensity, yielding damage-state-specific dispersions
   :math:`\beta_i`. This is the more realistic formulation and does not
   sacrifice the hierarchical property. Here :math:`k` independent binary
   probit regressions are fitted, one per damage state:

   .. math::

      P(\mathrm{DS} \geq ds_i \mid \mathrm{IM}) =
      \Phi(b_{0,i} + b_{1,i}\,\ln \mathrm{IM})

   Median and dispersion for each DS are recovered as
   :math:`\theta_i = \exp(-b_{0,i}/b_{1,i})` and
   :math:`\beta_i = 1/b_{1,i}`. After fitting, isotonic regression
   (Pool-Adjacent Violators Algorithm, PAVA) is applied to
   :math:`\ln(\theta_i)` to restore the hierarchical ordering constraint.
   Curves may cross in the tails but the ordering of 50th-percentile
   capacities is guaranteed.

   **Fragility curves**

   The probability of being in exactly damage state :math:`i` is:

   .. math::

      P(D = i \mid \mathrm{IM}) =
      P(D \geq i \mid \mathrm{IM}) - P(D \geq i+1 \mid \mathrm{IM})

   The exceedance fragility curves :math:`P(D \geq i \mid \mathrm{IM})`
   are obtained directly from the fitted cumulative model.

.. admonition:: References

   1. Jalayer, F., Ebrahimian, H., Trevlopoulos, K., and Bradley, B.
      (2023). Empirical tsunami fragility modelling for hierarchical
      damage levels. *Natural Hazards and Earth System Sciences*, 23(2),
      909–931. https://doi.org/10.5194/nhess-23-909-2023

   2. Nguyen, M. and Lallemant, D. (2022). Order Matters: The Benefits of
      Ordinal Fragility Curves for Damage and Loss Estimation. *Risk
      Analysis*, 42: 1136–1148. https://doi.org/10.1111/risa.13815

   3. Lallemant, D., Kiremidjian, A., and Burton, H. (2015). Statistical
      procedures for developing earthquake damage fragility curves.
      *Earthquake Engineering & Structural Dynamics*, 44, 1373–1389.
      https://doi.org/10.1002/eqe.2522

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.postprocessor import postprocessor

      pp = postprocessor()
      intensities = np.geomspace(0.05, 3.0, 50)
      damage_thresholds = [0.005, 0.015, 0.040, 0.080]

      # Constant dispersion — fully-ordered special case (default)
      # shape: (50, n_categories), where n_categories is the number of
      # distinct damage states observed (typically n_DS + 1 when DS=0
      # is present in the data)
      poes_const = pp.calculate_ordinal_fragility(
          imls=imls,
          edps=edps,
          damage_thresholds=damage_thresholds,
          intensities=intensities,
          dispersion_type='constant',
      )

      # Variable dispersion — general hierarchical case
      # shape: (50, len(damage_thresholds)) = (50, 4)
      poes_var = pp.calculate_ordinal_fragility(
          imls=imls,
          edps=edps,
          damage_thresholds=damage_thresholds,
          intensities=intensities,
          dispersion_type='variable',
      )
