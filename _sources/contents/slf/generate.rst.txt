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
       edp_range=np.linspace(0.001, 0.10, 100),
       grouping_flag=True,
       conversion=1.0,
       realizations=500,
       replacement_cost=1.0,
   )

   # 3. Generate and plot the SLF
   slf, cache = model.generate()
   plotter().plot_slf_model(
       slf, cache,
       edp_label="Interstorey Drift Ratio [-]",
       loss_label="Drift-Sensitive NSC Storey Loss Ratio",
       xlims=[0, 0.05],
       ylims=[0, 1],
       title="Drift-Sensitive SLF",
   )

.. admonition:: Theoretical Background

   A Storey Loss Function (SLF) links the expected repair loss at a given storey to the
   Engineering Demand Parameter (EDP) at that storey (Ramirez & Miranda, 2009;
   Shahnazaryan et al., 2021).

   **Component loss model**

   For each damageable component :math:`c` in performance group :math:`g`, the repair
   cost :math:`\ell_c` is a random variable that depends on the damage state
   :math:`DS_c`. Given the EDP level :math:`x`, the expected cost contribution is:

   .. math::

      E[\ell_c \mid x] = \sum_{i=1}^{n_{DS}}
        \mu_{c,i} \cdot P(DS_c = i \mid x)

   where :math:`\mu_{c,i}` is the mean repair cost for damage state :math:`i` and
   :math:`P(DS_c = i \mid x)` is derived from the component fragility functions via:

   .. math::

      P(DS_c = i \mid x) =
        P(DS_c \geq i \mid x) - P(DS_c \geq i+1 \mid x)

   **Monte Carlo sampling**

   Damage states are sampled via Monte Carlo simulation across a user-defined EDP range.
   For :math:`N` realisations, each realisation draws a damage state for every component
   at every EDP level. Component costs are summed within each performance group to obtain
   the total group loss ratio per realisation (loss normalised by ``replacement_cost``).

   **Empirical SLF percentiles**

   No parametric curve is fitted to the Monte Carlo loss cloud. Instead, at each EDP
   level the storey loss RATIO is summarised directly from the ``realizations`` Monte
   Carlo draws via its empirical 16th, 50th (median), and 84th percentiles:

   .. math::

      \hat{\ell}_g^{(p)}(x) = \text{Percentile}_p\left(\{\ell_g^{(n)}(x)\}_{n=1}^{N}\right),
      \quad p \in \{16, 50, 84\}

   where :math:`\ell_g^{(n)}(x)` is the simulated storey loss ratio for performance
   group :math:`g` at EDP level :math:`x` in realisation :math:`n`, and :math:`N` is
   ``realizations``. The median curve (:math:`p=50`) is the primary Storey Loss
   Function, returned as ``out[group]['slf']``; the 16th and 84th percentiles
   (``out[group]['slf_16th']`` / ``out[group]['slf_84th']``) bound the Monte Carlo
   scatter and are typically shown as a shaded band around the median curve (see
   :meth:`openquake.vmtk.plotter.plotter.plot_slf_model`).

   Because no functional form is imposed, the curves are exactly as smooth (or as
   noisy) as the underlying Monte Carlo sample — increasing ``realizations`` reduces
   sampling noise in the percentile curves without introducing any fitting bias.

   **Correlation trees**

   Components sharing a common support are rarely damaged independently of one
   another. A ``correlation_tree`` lets one component's damage state force a
   *minimum* damage state on another: whenever the *causation* component reaches (or
   exceeds) a specified damage state, the *dependent* component's simulated damage
   state is raised to at least a specified floor, before repair costs are computed.

   The tree is a ``pandas.DataFrame`` with one row per component and the following
   columns:

   - ``ID``: the component's own ``Component ID``. Every ``Component ID`` in
     ``component_data`` must have a matching row in the tree — the two are joined by
     ID, not by row order, so the tree may list components in any order.
   - ``DEPENDENT ON ITEM``: ``'Independent'`` for components with no forced
     dependency, or the causation component's ``Component ID`` (as a string)
     otherwise.
   - ``DS0``, ``DS1``, ..., ``DS{n}`` (one column per possible damage state of the
     causation component, including ``DS0`` = undamaged): for a dependent row,
     ``'Independent'`` below the triggering damage state and ``'DSk'`` (the forced
     floor) at and above it; for an independent row, ``'Independent'`` in every
     column.

   For example, interior door assemblies are typically framed into a partition
   wall's studs — once the partition passes cosmetic cracking (DS2), the door frame
   racks along with it. If component 5 (the partition, 4 damage states) causes
   component 3 (the door) to be forced to at least DS1:

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
          grouping_flag=True,
          realizations=500,
      )

   ``slfgenerator`` validates the tree at construction time: every ``Component ID``
   in the inventory must have a matching row in the tree, and no forced damage state
   may exceed the number of damage states defined for that component. Both
   ``component_data['Component ID']`` and ``correlation_tree['ID']`` must be unique
   positive integers — the class indexes fragilities, damage states, and repair costs
   internally by ``Component ID``, so duplicate IDs raise a ``ValueError`` at
   construction time rather than silently mismatching components.

