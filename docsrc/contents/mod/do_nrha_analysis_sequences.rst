Nonlinear Time-History Analysis of Record Sequences
===================================================

.. automethod:: openquake.vmtk.modeller.modeller.do_nrha_analysis_sequences

.. admonition:: Theoretical Background

   ``do_nrha_analysis`` takes a fixed time step. This method instead takes an
   explicit ``time_vector``, so that records with different sampling rates can
   be concatenated into a single input (for example a mainshock followed by an
   aftershock, separated by zero-acceleration padding) and the structure is
   integrated through the whole sequence, carrying its accumulated damage from
   one record to the next.

   **Record detection.** Individual records are located automatically as the
   stretches between *quiescent* zones, where the absolute acceleration stays
   below ``quiescence_threshold`` for at least ``padding_duration`` seconds.
   Their time windows are returned in ``sequence_boundaries``.

   **Outputs.** Peak drifts, accelerations and hysteretic energies are
   reported for the full sequence and for each record
   (``peak_drift_per_sequence``, ``max_peak_drift_per_sequence``,
   ``hysteretic_energy_per_storey_per_sequence``, ...), so the damage
   accumulated by the second shock can be separated from the first.
   Hysteretic energy is the dissipated energy only (signed force–velocity
   integration), and collapse is detected by both the MinMax material response
   and an independent nodal-displacement check.

.. admonition:: Example
   :class: note

   .. code-block:: python

      import numpy as np

      m.compile_model()
      m.do_gravity_analysis()
      m.do_modal_analysis(num_modes=2)

      # ``sequence.txt``: records separated by >= 40 s of zero acceleration;
      # ``time.txt``: the matching (possibly irregular) time values [s]
      time_vector = np.loadtxt("time.txt")
      results = m.do_nrha_analysis_sequences(
          fnames=["sequence.txt"],
          time_vector=time_vector,
          sf=9.81,                  # records in g
          padding_duration=40.0,
          drift_thresholds=[0.0015, 0.0030, 0.0045, 0.0135],
          save_animation_path=None,
      )
