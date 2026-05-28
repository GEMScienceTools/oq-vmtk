# Incremental Dynamic Analysis Demo

This Jupyter Notebook presents an example of Incremental Dynamic Analysis carried out on a multi-degree-of-freedom stick model using the `modeller` module.  To do so, the "Hunt, Trace and Fill" algorithm is employed to scale ground motion records until structural collapse is observed. Once structural collapse or dynamic instability is observed, the IDA curve is traced back and filled with smaller increments of scaling factor.

From these scaled results, global response quantities, such as peak storey drifts, are extracted across a range of increasing intensity levels and the `postprocessor` module is then applied to derive fragility functions by fitting lognormal cumulative distribution functions based on the observed dynamic capacity (i.e., IDA curve) at specific demand-based damage states.

NOTE: Throughout the notebook, a demonstration of managing I/O in OQ-VMTK is presented. This is particularly valuable because it ensures that OQ-VMTK’s modules receive input data in a consistent, ready-to-use format. Users are encouraged to follow this procedure for efficient data processing and analysis.