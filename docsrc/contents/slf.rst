Storey Loss Function Generation
################################

The ``slfgenerator`` class generates Storey Loss Functions (SLFs) from a component
inventory in the FEMA P-58 format: quantities, fragility functions, damage state
hierarchies and repair-cost distributions. An SLF gives, for the components of one
storey that are sensitive to one Engineering Demand Parameter (EDP: peak storey drift
or peak floor acceleration), the repair cost as a fraction of their replacement cost,
as a function of that EDP.

For every EDP value the generator returns the mean, standard deviation and
16th/50th/84th percentiles of the storey loss ratio, estimated by Monte Carlo
simulation. The simulation follows FEMA P-58: every component (or, where its units are
not correlated, every unit) has its own capacity; mutually exclusive and simultaneous
damage states are sampled explicitly; unit repair costs are uncertain. Each realisation
is divided by the replacement cost of the same components at the same unit costs, so
the loss ratio lies in [0, 1] and reaches 1 when everything is replaced.
An optional ``correlation_tree`` lets one component's damage force a minimum damage
state on another (e.g. a partition wall and the door framed into it).

.. toctree::

   slf/init
   slf/generate

References
----------

1. Ramirez, C. and Miranda, E., (2009) "Building-specific loss estimation methods and tools for simplified
   performance-based earthquake engineering", John A. Blume Earthquake Engineering Center, Stanford University.

2. FEMA (2018) "Seismic Performance Assessment of Buildings, Volume 1 – Methodology", FEMA P-58-1,
   2nd edition, Federal Emergency Management Agency, Washington, D.C.

3. Shahnazaryan, D., O'Reilly, G.J., Monteiro R. "Story loss functions for seismic design and assessment:
   Development of tools and application," Earthquake Spectra 2021; 37(4): 2813-2839.
   DOI: 10.1177/87552930211023523.

4. Shahnazaryan, D., O'Reilly, G.J., Monteiro R. "Development of a Python-Based Storey Loss Function
   Generator," COMPDYN 2021 - 8th International Conference on Computational Methods in Structural Dynamics
   and Earthquake Engineering, 2021. DOI: 10.7712/120121.8659.18567.

5. Papadopoulos, A.N., Vamvatsikos, D., Kazantzi, A.K. "Development and Application of FEMA P-58
   Compatible Story Loss Functions," Earthquake Spectra 2019; 35(1): 95-112. DOI: 10.1193/102417EQS222M.
