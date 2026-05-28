# Intensity Measure Selection Demo

This Jupyter Notebook demonstrates a framework for selecting optimal intensity measures (IMs) for seismic demand modelling using the `postprocessor` module. Four IM selection metrics are evaluated: Efficiency, Proficiency, Practicality, and the Relative Sufficiency Measure (RSM).

The notebook is organised into three parts:
- **Part I**: MCA-based IM selection — evaluating IM candidates against Modified Cloud Analysis results.
- **Part II**: IDA-based IM selection — evaluating the same candidates against Incremental Dynamic Analysis results.
- **Part III**: MCA vs. IDA comparison — assessing whether IM rankings are consistent across analysis methods.

The RSM sign convention and key caveats for interpreting results across analysis types are discussed throughout.
