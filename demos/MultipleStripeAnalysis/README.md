# Multiple Stripe Analysis Demo

This Jupyter Notebook presents an example of Multiple Stripe Analysis (MSA) carried out on a multi-degree-of-freedom (MDOF) stick model using the `modeller` module. Ground motion records are organised into distinct intensity-based "stripes" at hazard-consistent IM levels. Global response quantities, such as peak storey drifts and peak floor accelerations, are extracted across each stripe.

From these results, the `postprocessor` module is applied to derive fragility functions using Maximum Likelihood Estimation (MLE) at specific demand-based damage states.

A vulnerability, or loss model, is subsequently developed by combining these fragility-based probabilities of exceedance with a deterministic consequence model through damage-to-loss ratios. To account for the inherent uncertainty in losses at a given shaking intensity, a beta distribution is utilized.

Finally, visual outputs using the `plotter` module include stripe scatter plots, seismic demand profiles, fragility functions, and vulnerability models.

NOTE: As an additional feature, the notebook demonstrates how to export NLTHA response quantities. This step is valuable because it ensures that OQ-VMTK's postprocessor and plotter modules receive input data in a consistent, ready-to-use format. Users are encouraged to follow this procedure for efficient data processing and analysis.
