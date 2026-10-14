SLF Generation
==============

.. automethod:: openquake.vmtk.slfgenerator.slfgenerator.generate

Self-contained example
----------------------

The snippet below is fully runnable from the repository root after
``pip install .`` — it uses the inventory CSVs shipped with the demo
notebook.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from openquake.vmtk.slfgenerator import slfgenerator
   from openquake.vmtk.plotter import plotter

   # 1. Load a drift-sensitive component inventory
   inventory = pd.read_csv(
       "demos/StoreyLossFunctionGeneration/in/inventory_psd.csv"
   )

   # 2. Configure the generator
   model = slfgenerator(
       component_data=inventory,
       edp="PSD",
       edp_range=np.linspace(0.0, 0.10, 101),
       grouping_flag=True,
       conversion=1.0,
       realizations=2000,
       seed=2026,
   )

   # 3. Generate and plot the SLF
   slf, cache = model.generate()
   group = slf["PSD, NS"]
   print(group["replacement_cost"], group["slf"][50], group["slf_cov"][50])

   plotter().plot_slf_model(
       slf, cache,
       edp_label="Interstorey Drift Ratio [-]",
       loss_label="Drift-Sensitive NSC Storey Loss Ratio",
       xlims=[0, 0.05],
       ylims=[0, 1],
       title="Drift-Sensitive SLF",
   )

Each record of ``slf`` holds:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Content
   * - ``edp_range``
     - EDP values.
   * - ``slf``
     - Mean storey loss ratio: the Storey Loss Function.
   * - ``slf_std``, ``slf_cov``
     - Standard deviation of the loss ratio and ``slf_std / slf``.
   * - ``slf_16th``, ``slf_50th``, ``slf_84th``
     - Empirical percentiles of the loss ratio.
   * - ``replacement_cost``
     - Expected replacement cost of the group's components (or the
       ``replacement_cost`` argument). By default each realisation is divided by
       the replacement cost at its own unit costs, so the loss ratio lies in [0, 1].

To combine the drift- and acceleration-sensitive SLFs of a storey into one
nonstructural loss ratio, weight each by its share of the replacement cost:
:math:`L = w_d L_d + w_a L_a` with :math:`w_d = RC_d / (RC_d + RC_a)`.

.. admonition:: Theoretical Background

   A Storey Loss Function (SLF) links the expected repair loss of the components of a
   storey to the Engineering Demand Parameter (EDP) :math:`x` at that storey (Ramirez &
   Miranda, 2009; Papadopoulos et al., 2019; Shahnazaryan et al., 2021). The component
   model is that of FEMA P-58.

   **Fragility of a limit state**

   Each component has one or more limit states, in sequence; limit state :math:`k` has
   a lognormal fragility with median :math:`\theta_k` and dispersion :math:`\beta_k`:

   .. math::

      P(LS \geq k \mid x) = \Phi\left(\frac{\ln(x / \theta_k)}{\beta_k}\right)

   with the curves kept from crossing, :math:`P(LS \geq k+1 \mid x) \leq P(LS \geq k
   \mid x)`, and zero at :math:`x = 0`.

   **Damage states of a limit state**

   A limit state holds one damage state, or several that are either *mutually
   exclusive* (exactly one occurs, damage state :math:`i` with probability
   :math:`w_i`, :math:`\sum w_i = 1`) or *simultaneous* (each occurs independently
   with probability :math:`p_i`, conditioned on at least one occurring). The expected
   unit repair cost given limit state :math:`k` is

   .. math::

      \bar c_k = \bar c_i \quad \text{(single)}, \qquad
      \bar c_k = \sum_i w_i \, \bar c_i \quad \text{(MutEx)}, \qquad
      \bar c_k = \frac{\sum_i p_i \, \bar c_i}{1 - \prod_i (1 - p_i)} \quad \text{(Simul)}

   where :math:`\bar c_i` is the mean unit cost of damage state :math:`i`:
   :math:`\theta_c \exp(\beta_c^2 / 2)` for a lognormal cost with median
   :math:`\theta_c`, and :math:`\mu \left[1 + \nu \, \varphi(1/\nu) / \Phi(1/\nu)\right]`
   for a normal cost with mean :math:`\mu` and coefficient of variation :math:`\nu`
   truncated at zero.

   **Monte Carlo simulation**

   The SLF is estimated from ``realizations`` Monte Carlo realisations. In each one:

   - every component draws one capacity (a uniform number :math:`u`) held over the
     whole EDP range, and is in limit state
     :math:`\#\{k : u < P(LS \geq k \mid x)\}` at EDP :math:`x`; components are
     independent of one another unless a correlation tree links them. Where the
     units of a component are not correlated (``Correlated`` = ``NO``), each unit
     draws its own capacity;
   - the damage state within a mutually exclusive or simultaneous limit state is
     drawn once per unit and limit state;
   - the unit repair costs of a component are drawn at one probability level,
     shared by its damage states and its units: one price level for repairing and
     replacing the same item. A simultaneous outcome costs the sum of its damage
     states;
   - the unit replacement cost :math:`RC_c^{(n)}` of the realisation is the cost of
     the replacement outcome (the damage outcome with the largest expected cost) at
     that price level, scaled to ``Replacement Cost`` when given; no unit is
     repaired for more than it.

   Each component draws from its own random stream, so with the same ``seed`` a
   model with and without a correlation tree share their random numbers and the
   effect of the tree can be compared realisation by realisation.

   **Storey loss ratio**

   The loss ratio of realisation :math:`n` is the repair cost of the group's
   components divided by their replacement cost at the same unit costs:

   .. math::

      L^{(n)}(x) = \frac{C^{(n)}(x)}{\sum_c Q_c \, RC_c^{(n)}}

   It lies in [0, 1], equals 1 when every unit is in its replacement outcome, and
   does not change with the price level of a component, which scales its repair and
   replacement alike. The SLF is its mean, ``slf``; ``slf_std``, ``slf_cov`` =
   ``slf_std`` / ``slf`` and the empirical 16th/50th/84th percentiles (``slf_16th``,
   ``slf_50th``, ``slf_84th``) complete it. Because every realisation lies in
   [0, 1], the mean and standard deviation are always those of a distribution on
   [0, 1] (e.g. a Beta can be fitted to them).

   With ``replacement_cost`` given, every realisation is divided by that fixed cost
   instead; the ratio then keeps the price-level uncertainty and is not bounded by 1.

   **Exact expectations**

   The probability of each limit state after correlation tree forcing and the
   expected repair cost of each component are also computed exactly:

   .. math::

      E[C_c \mid x] = Q_c \sum_{k \geq 1} P(LS_c = k \mid x) \, \bar c_{c,k},
      \qquad P(LS_c = k \mid x) = P(LS_c \geq k \mid x) - P(LS_c \geq k+1 \mid x)

   (``cache[group]['limit_state_probabilities']`` and
   ``['expected_component_loss']``). Their sum over the expected replacement cost,
   ``cache[group]['ratio_of_expectations']``, is a check on ``slf``: the mean of a
   ratio and the ratio of the means differ, by about 0.01 at most.

   Not modelled: economies of scale (unit costs do not depend on the number of
   damaged units).

   **Correlation trees**

   Components sharing a common support are rarely damaged independently of one
   another. A ``correlation_tree`` lets one component's damage force a *minimum*
   limit state on another: whenever the *causation* component is in limit state
   :math:`k`, the *dependent* component is raised to at least limit state
   :math:`f(k)`. Its own capacity still governs above that floor:

   .. math::

      P(LS_d \geq j \mid x) = 1 - \big[1 - P_{own}(LS_d \geq j \mid x)\big]
        \big[1 - P(LS_c \geq t_j \mid x)\big], \qquad t_j = \min\{k : f(k) \geq j\}

   Where units are sampled individually, dependent unit :math:`b` of :math:`n_d` is
   hosted by causation unit :math:`\lfloor b \, n_c / n_d \rfloor` of :math:`n_c`, so
   that each causation unit carries an equal share of the dependent units (e.g. the
   light fixtures of a ceiling panel fall with that panel).

   The tree is a ``pandas.DataFrame`` with one row per component and the columns:

   - ``ID``: the component's own ``Component ID``. Every ``Component ID`` in
     ``component_data`` must have a matching row — the two are joined by ID, not by
     row order.
   - ``DEPENDENT ON ITEM``: ``'Independent'``, or the causation component's
     ``Component ID``.
   - ``DS0``, ``DS1``, ..., ``DS{n}``: for a dependent row, column ``DSk`` gives the
     limit state (``'DSj'``) forced on the dependent when the causation component is
     in limit state ``k`` (``DS0`` = undamaged), or ``'Independent'``. For
     components whose damage states are all sequential (the default) limit state
     ``k`` is damage state ``k``. Columns beyond the causation component's limit
     states are ignored.

   For example, interior door assemblies are framed into a partition wall's studs —
   once the partition passes cosmetic cracking (DS2), the door frame racks along with
   it. If component 5 (the partition, 4 damage states) forces component 3 (the door)
   to at least DS1:

   .. code-block:: python

      import pandas as pd

      correlation_tree = pd.DataFrame({
          "ID":                [1, 2, 3, 4, 5, 6, 7],
          "DEPENDENT ON ITEM": ["Independent", "Independent", "5",
                                 "Independent", "Independent",
                                 "Independent", "Independent"],
          "DS0": ["Independent"] * 7,
          "DS1": ["Independent"] * 7,
          "DS2": ["Independent", "Independent", "DS1", "Independent",
                  "Independent", "Independent", "Independent"],
          "DS3": ["Independent", "Independent", "DS1", "Independent",
                  "Independent", "Independent", "Independent"],
          "DS4": ["Independent", "Independent", "DS1", "Independent",
                  "Independent", "Independent", "Independent"],
      })

      model = slfgenerator(
          component_data=inventory,
          edp="PSD",
          correlation_tree=correlation_tree,
          realizations=2000,
          seed=2026,
      )

   ``slfgenerator`` validates the tree at construction time. It raises a
   ``ValueError`` if a ``Component ID`` has no row, an ID is duplicated, a causation
   component does not exist or is the component itself, the dependencies form a
   cycle, a forced limit state exceeds the dependent's limit states, an undamaged
   causation component (``DS0``) forces damage, or the forced limit state decreases
   as the causation component gets more damaged. Chains (a dependent that is itself
   a causation component) are allowed.
