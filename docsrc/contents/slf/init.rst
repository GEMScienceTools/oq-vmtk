Initialisation
==============

.. automethod:: openquake.vmtk.slfgenerator.slfgenerator.__init__

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      import pandas as pd
      from openquake.vmtk.slfgenerator import slfgenerator

      inventory = pd.read_csv("demos/StoreyLossFunctionGeneration/in/inventory_psd.csv")
      model = slfgenerator(
          component_data=inventory,
          edp="PSD",
          edp_range=np.linspace(0.0, 0.10, 101),
          grouping_flag=True,
          conversion=1.0,
          realizations=2000,
          seed=2026,
      )

Component inventory
-------------------

One row per component. The columns follow the FEMA P-58 fragility and consequence
database, so values can be copied from it.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Column
     - Content
   * - ``Component ID``
     - Unique positive integer (need not be sequential; blank IDs are numbered
       ``1..n``).
   * - ``Description``
     - Optional text.
   * - ``EDP``
     - ``PSD`` (or ``IDR``) or ``PFA``.
   * - ``Typology``
     - ``S`` (structural) or ``NS`` (nonstructural). ``(EDP, Typology)`` forms the
       default performance groups.
   * - ``Performance Group``
     - Optional; more than one value overrides the default grouping.
   * - ``Quantity``
     - Number of costing units of the component on the storey.
   * - ``Damage States``
     - Number of damage states ``n``.
   * - ``DSk, Median``, ``DSk, Total Dispersion``
     - Median (in EDP units) and logarithmic standard deviation of the lognormal
       fragility of the limit state that damage state ``k`` belongs to.
   * - ``DSk, Cost``, ``DSk, Cost Dispersion``, ``DSk, Best Fit``
     - Unit repair cost in damage state ``k``. ``Lognormal``: median and
       logarithmic standard deviation. ``Normal`` (default when blank): mean and
       coefficient of variation, truncated at zero.
   * - ``DS Hierarchy``
     - Optional FEMA P-58 hierarchy, e.g. ``Seq(DS1,DS2)``,
       ``Seq(DS1,MutEx(DS2,DS3))``, ``Simul(DS1,DS2,DS3,DS4)``. Each item of the
       sequence is one limit state with one fragility. Default: all damage states
       sequential.
   * - ``DSk, Probability``
     - Optional. Within a ``MutEx`` limit state, the probability that damage state
       ``k`` is the one that occurs (they sum to one); within a ``Simul`` limit
       state, the probability that it occurs, independently of the others.
   * - ``Correlated``
     - Optional FEMA P-58 "Correlated?" flag. ``YES`` (default): all units of the
       component share one damage state. ``NO``: every unit is damaged
       independently; ``Quantity`` must then count the units on the storey (it is
       split into ``ceil(Quantity)`` equal blocks).
   * - ``Replacement Cost``
     - Optional cost of replacing one unit. Default: the largest expected repair
       cost of any damage outcome of the component.

The damage states of a limit state share its fragility: the median and dispersion are
given for its first damage state and may be left blank or repeated for the others.
Medians must not decrease from one limit state to the next. Other columns are ignored.

The fragility FEMA P-58 publishes for components in a Simultaneous hierarchy (e.g.
traction elevators, D1014.011) is the probability that at least one of their damage
states occurs; enter it as one ``Simul`` limit state.

Damage state hierarchies
------------------------

The hierarchy of each component is set by its ``DS Hierarchy`` cell, and the
probabilities of grouped damage states by its ``DSk, Probability`` cells. The demo
inventories (``demos/StoreyLossFunctionGeneration/in/``) contain one example of each
case; their relevant columns are:

.. list-table::
   :header-rows: 1
   :widths: 30 22 9 9 9 9 12

   * - Component (FEMA P-58 ID)
     - ``DS Hierarchy``
     - ``DS1, Median``
     - ``DS2, Median``
     - ``DS3, Median``
     - ``DS4, Median``
     - Probabilities
   * - Stairs (C2011.001b), sequential
     - ``Seq(DS1,DS2,DS3)``
     - 0.005
     - 0.017
     - 0.028
     -
     - blank
   * - Air handling unit, unanchored (D3052.011a), mutually exclusive
     - ``MutEx(DS1,DS2)``
     - 0.25
     - 0.25
     -
     -
     - 0.67, 0.33
   * - Partition on wood studs (C1011.011a), sequential then mutually exclusive
     - ``Seq(DS1,MutEx(DS2,DS3))``
     - 0.0021
     - 0.0071
     - 0.0071
     -
     - 1.0, 0.8, 0.2
   * - Traction elevator (D1014.011), simultaneous
     - ``Simul(DS1,DS2,DS3,DS4)``
     - 0.39
     - 0.39
     - 0.39
     - 0.39
     - 0.26, 0.79, 0.68, 0.17

- **Sequential**: limit state ``k`` must be reached before ``k + 1``; one fragility
  per damage state. This is the default when ``DS Hierarchy`` is blank.
- **Mutually exclusive**: once the limit state is reached, exactly one of its damage
  states occurs, with the given probabilities (summing to one) — e.g. the air
  handling unit either needs repair (DS1, 0.67) or is beyond repair (DS2, 0.33).
- **Simultaneous**: once the limit state is reached, each of its damage states
  occurs independently with its probability, at least one of them occurring; the
  repair costs of the damage states that occur add up.
- The groups can be combined in a sequence, e.g. the partition cracks first (DS1),
  then at a higher drift either buckles (DS2, 0.8) or is crushed (DS3, 0.2).

In CSV form, the elevator row reads (cost columns omitted)::

   Component ID,Description,EDP,...,Damage States,"DS1, Median",...,DS Hierarchy,"DS1, Probability","DS2, Probability","DS3, Probability","DS4, Probability",Correlated
   5,Elevators,PFA,...,4,0.39,...,"Simul(DS1,DS2,DS3,DS4)",0.26,0.79,0.68,0.17,YES

A correlation tree refers to limit states: for the partition above, ``DS2`` in a tree
means its second limit state (DS2 or DS3).
