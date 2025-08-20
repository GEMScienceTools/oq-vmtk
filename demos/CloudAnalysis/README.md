# Cloud Analysis Demo

This Jupyter Notebook presents an example of nonlinear dynamic analysis, focusing on cloud analysis of a multi-degree-of-freedom (MDOF) stick model. Both global and local response quantities, such as peak storey drifts and peak floor accelerations, are extracted. A probabilistic seismic demand model (PSDM) is then fitted to the intensity measure (IM) and engineering demand parameter (EDP) data using linear regression in log–log space. From this model, fragility functions are derived for arbitrary demand-based damage states. A vulnerability, or loss model, is subsequently developed by combining the fragility-based probabilities of exceedance with a deterministic consequence model through damage-to-loss ratios, while also accounting for uncertainty in losses at a given shaking intensity using a beta distribution. Visual outputs of the PSDM, seismic demand profiles, fragility functions, and vulnerability models are included.

The MDOF stick model is implemented in OpenSees through the  `modeller` module. This involves defining nodes, masses, elements, and storey-level force–deformation relationships represented with nonlinear springs (i.e., zero-length elements). A single ground motion record is used to demonstrate nonlinear time-history analysis (NLTHA) within the  `modeller` framework.

The `postprocessor` module is then applied to establish the IM–EDP relationships and to derive the fragility and vulnerability models.

Finally, the `plotter` module is used to produce visualizations of the PSDM fit to the IM–EDP dataset, seismic demand profiles (peak storey drift and peak floor acceleration along the model height), and the fragility and vulnerability functions.

NOTE: As an additional feature, the notebook demonstrates how to export NLTHA response quantities. This step is valuable because it ensures that OQ-VMTK’s postprocessor and plotter modules receive input data in a consistent, ready-to-use format. Users are encouraged to follow this procedure for efficient data processing and analysis.
