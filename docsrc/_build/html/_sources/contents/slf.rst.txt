Storey Loss Function Generation
################################

The ``slf_generator`` class generates Storey Loss Functions (SLFs) from fragility,
consequence, and quantity data. SLFs link expected repair loss at a storey to the
Engineering Demand Parameter (EDP). A Monte Carlo approach is used to sample damage
states and associated repair costs across a user-defined inventory of damageable
components.

.. toctree::

   slf/init
   slf/generate

References
----------

1. Ramirez, C. and Miranda, E., (2009) "Building-specific loss estimation methods and tools for simplified
   performance-based earthquake engineering", John A. Blume Earthquake Engineering Center, Stanford University.

2. Shahnazaryan, D., O'Reilly, G.J., Monteiro R. "Story loss functions for seismic design and assessment:
   Development of tools and application," Earthquake Spectra 2021; 37(4): 2813-2839.
   DOI: 10.1177/87552930211023523.

3. Shahnazaryan, D., O'Reilly, G.J., Monteiro R. "Development of a Python-Based Storey Loss Function
   Generator," COMPDYN 2021 - 8th International Conference on Computational Methods in Structural Dynamics
   and Earthquake Engineering, 2021. DOI: 10.7712/120121.8659.18567.
