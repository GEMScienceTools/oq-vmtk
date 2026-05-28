# Fragility Analysis Demo

This Jupyter Notebook demonstrates the different fragility function derivation functionalities available in OQ-VMTK and illustrates the effect of each method on subsequent downstream applications. All fragility fitting methods are applied to the same pre-computed Modified Cloud Analysis (MCA) dataset using the `postprocessor` module, covering lognormal cumulative distribution function variants, generalised linear models (GLM logit and probit), ordinal cumulative link models (CLM), and Markov Chain Monte Carlo (MCMC) sampling.

The downstream effect of each fragility model choice on loss modelling and risk metrics — including mean vulnerability functions, average annual damage probabilities, and average annual loss ratios — is illustrated throughout.
