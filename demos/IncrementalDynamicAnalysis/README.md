# Incremental Dynamic Analysis Demo

This Jupyter Notebook presents an example of Incremental Dynamic Analysis (IDA) carried out on a multi-degree-of-freedom (MDOF) stick model using the `modeller` module. Both global and local response quantities, such as peak storey drifts and peak floor accelerations, are extracted across a range of increasing intensity levels. To do so, the "Hunt, Trace and Fill" algorithm is employed to scale ground motion records until structural collapse is observed. Once structural collapse or dynamic instability is observed, the IDA curve is traced back and filled with smaller increments of scaling factor.

From these scaled results, the `postprocessor` module is then applied to derive fragility functions by fitting lognormal cumulative distribution functions based on the observed dynamic capacity (i.e., IDA curve) at specific demand-based damage states.

A vulnerability, or loss model, is subsequently developed by combining these fragility-based probabilities of exceedance with a deterministic consequence model through damage-to-loss ratios. To account for the inherent uncertainty in losses at a given shaking intensity, a beta distribution is utilized.

Finally, visual outputs using the the `plotter` module, include IDA curves, seismic demand profiles, fragility functions, and vulnerability models.

NOTE: As an additional feature, the notebook demonstrates how to export NLTHA response quantities. This step is valuable because it ensures that OQ-VMTK’s postprocessor and plotter modules receive input data in a consistent, ready-to-use format. Users are encouraged to follow this procedure for efficient data processing and analysis.
