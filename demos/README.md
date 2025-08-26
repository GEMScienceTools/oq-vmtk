# Demos

This folder contains an ensemble of example python notebooks demonstrating various steps within the GEM global vulnerability workflow highlighting earthquake catastrophe modelling and visualization techniques. Below is an overview of the four available examples:

## IntensityMeasureProcessing:  Example of Ground-Motion Record Processing
This example provides a workflow for processing intensity measures (IM) from ground-motion records using the `IMCalculator` module. This includes  spectral analysis and identification of ground shaking intensities for distinct IM types (e.g., PGA, Arias Intensity, FIV3, etc.).

## ModelCompilation: Example of OpenSees Model Compilation
This example provides a workflow for compiling simplified single-degree- and multi-degree-of freedom systems in OpenSeesPy using the `modeller` module and a basic set of input arguments. This example also includes the application of the `calibration` module to characterise storey-based force-deformation relationships for MDOF stick-and-mass systems based on SDOF capacity and parameters

## PushoverAnalysis: Example of Static and Cyclic Pushover Analysis
This example provides an application of nonlinear static procedures such as monotonic static and cyclic pushover analyses on a multi-degree-of-freedom (MDOF) stick model where global and local response parameters such as base shear, storey displacements and interstorey drifts are quantified.
The MDOF stick model is compiled and analysed in OpenSees using the `modeller` module.

## NonlinearTimeHistoryAnalysis: Example of Nonlinear Time-History Analysis
This example provides an application of nonlinear dynamic procedures such as nonlinear time-history on a multi-degree-of-freedom (MDOF) stick model where global and local response parameters such as storey displacements, interstorey drifts and floor accelerations are quantified.
The MDOF stick model is compiled and analysed in OpenSees using the `modeller` module and the seismic demands (i.e., drifts and accelerations) are visualised using the `plotter` module. Useful note on post-processing analysis quantities is also demonstrated to provide users with a consistent output readable across all OQ-VMTK modules and functions.

## FragilityAnalysis: Example of Supported Fragility Analyses Methods
This example demonstrates the application of the `postprocessor` module to derive fragility functions through both conventional and state-of-the-art methodologies. It applies multiple approaches to estimate probabilities of exceedance for arbitrary demand-based damage states following nonlinear time-history analyses. The methods include lognormal cumulative distribution functions, generalized linear models, and ordinal models. The treatment of fragility function rotation and additional epistemic uncertainty in fragility functions is also addressed.

## StoreyLossFunctionGeneration: Example of Storey Loss Function Generation
This example demonstrates an application for reading inventory files that list damageable structural components, nonstructural components, and building contents, along with their associated metrics—such as quantities, fragility function parameters based on demand, and corresponding costs. The core of this application is the use of the `SLFGenerator` module, which creates storey loss functions. These functions link expected decision variables, whether financial loss, societal impact, or downtime, to seismic demand, represented by engineering demand parameters (e.g., peak storey drift or peak floor acceleration).

## StoreyLossFunctionApplication: Example of Storey Loss Function Application
This example demonstrates an application of storey loss functions where vulnerability functions associated with nonstructural components are derived. The loss model is derived after interpolating the storey loss associated with engineering demand parameters (i.e., peak storey drift or peak floor acceleration) quantified following nonlinear time-history analyses.

## CloudAnalysis: Example of End-to-End Vulnerability Analysis of an MDOF System using Cloud Analysis
This example performs nonlinear time-history analysis on a multi-degree-of-freedom structural system calibrated using SDOF global capacity and using cloud analysis to assess seismic performance. The example additionally demonstrates postprocessing of cloud analysis results, including fragility curve estimation and vulnerability assessment visualization. It incorporates multiple modules including the `calibration`, `modeller`, `postprocessor`, `plotter` and `utilities` modules

---

Each example includes relevant scripts, input data, and instructions to guide users through the analysis. Feel free to explore and modify them to suit your specific needs!
