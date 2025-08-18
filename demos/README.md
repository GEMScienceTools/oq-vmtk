# Demos

This folder contains an ensemble of example python notebooks demonstrating various steps within the GEM global vulnerability workflow highlighting earthquake catastrophe modelling and visualization techniques. Below is an overview of the four available examples:

## IntensityMeasureProcessing:  Example of Ground-Motion Record Processing
This example provides a workflow for processing ground-motion records, such as spectral analysis and identification if intensity measure values for distinct intensity measure types (e.g., PGA, Arias Intensity, etc.).

## PushoverAnalysis: Example of Static and Cyclic Pushover Analysis
This example provides an application of nonlinear static procedures such as monotonic static and cyclic pushover analyses on a multi-degree-of-freedom (MDOF) stick model where global and local response parameters such as base shear, storey displacements and interstorey drifts are quantified.
The MDOF stick model is compiled and analysed in OpenSees using the `modeller` module.

## Example 2: End-to-End Vulnerability Analysis of an MDOF System using Cloud Analysis
This example performs nonlinear time-history analysis on a multi-degree-of-freedom (MDOF) structural system calibrated using SDOF global capacity and using cloud analysis to assess seismic performance. The example additionally demonstrates postprocessing of cloud analysis results, including fragility curve estimation and vulnerability assessment visualization.

## Example 3: Generating and Visualizing Storey Loss Functions
This example focuses on computing and visualizing storey loss functions, providing insights into seismic damage assessment and economic loss estimation.

Each example includes relevant scripts, input data, and instructions to guide users through the analysis. Feel free to explore and modify them to suit your specific needs!
