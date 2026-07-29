Ordinal Fragility Functions
===========================

.. automethod:: openquake.vmtk.postprocessor.postprocessor.calculate_ordinal_fragility

.. admonition:: Theoretical Background


   The ordinal regression approach treats damage state assignment as an ordered
   categorical outcome, jointly fitting all damage states in a single model while
   respecting their natural ordering (Nguyen & Lallemant, 2022).

   **Ordered response model**

   Let :math:`D \in \{0, 1, \ldots, k\}` be the observed damage state (0 = no damage,
   :math:`k` = highest damage state). The cumulative exceedance probability for
   damage state :math:`i` is:

   .. math::

      P(D \geq i \mid \text{IM}) = g^{-1}(\alpha_i - \beta \ln \text{IM}),
      \quad i = 1, \ldots, k

   where :math:`\alpha_i` are ordered thresholds satisfying
   :math:`\alpha_1 \leq \alpha_2 \leq \cdots \leq \alpha_k`,
   :math:`\beta` is a shared slope common to all damage states, and
   :math:`g` is a link function (logit or probit).

   **Shared slope constraint**

   Unlike fitting each damage state independently, ordinal regression enforces a
   single slope :math:`\beta` across all damage states. This prevents fragility
   curve crossings and improves statistical efficiency by using all damage
   observations simultaneously.

   **Fragility curves**

   The probability of being in exactly damage state :math:`i` is:

   .. math::

      P(D = i \mid \text{IM}) =
      P(D \geq i \mid \text{IM}) - P(D \geq i+1 \mid \text{IM})

   The exceedance fragility curves :math:`P(D \geq i \mid \text{IM})` are obtained
   directly from the fitted cumulative model.

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np
      from openquake.vmtk.postprocessor import postprocessor

      pp = postprocessor()
      intensities = np.geomspace(0.05, 3.0, 50)
      # imls: 1-D array of IM levels; damage_states: integer array (0, 1, 2, …)
      poes = pp.calculate_ordinal_fragility(
          imls=imls,
          damage_states=damage_states,
          intensities=intensities,
      )
