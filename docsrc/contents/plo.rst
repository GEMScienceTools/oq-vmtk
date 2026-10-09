Visualisation
#############

The ``plotter`` class creates publication-quality plots for all ``oq-vmtk``
analysis outputs. It covers modal shape plots, cloud/MCA scatter plots, IDA fan
plots, MSA stripe plots, fragility curves, vulnerability functions, Storey Loss
Function outputs, and animated NRHA responses. All plots share a consistent style
(fonts, line widths, colour schemes) and can optionally be saved to disk.

All single-panel static plots (MCA, IDA, MSA, fragility, SLF and vulnerability
plots) use the same figure size and fixed axes margins, so the plotting grid has
identical dimensions across plots regardless of labels, legends or secondary
axes. The multi-panel demand-profile and modal-shape plots, and the animations,
keep their own layouts.

.. toctree::

   plo/plot_modes
   plo/plot_demand_profiles
   plo/animate_spo
   plo/animate_cpo
   plo/animate_nrha
   plo/plot_mca_analysis
   plo/plot_ida_analysis
   plo/plot_msa_analysis
   plo/plot_fragility_from_mca
   plo/plot_fragility_from_ida
   plo/plot_fragility_from_msa
   plo/plot_slf_model
   plo/plot_vulnerability_function
