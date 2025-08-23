# Storey Loss Function Application Demo

This Jupyter Notebook demonstrates an application of storey loss functions (SLFs) to derive a building-class vulnerability model. Using pre-existing SLFs, the workflow focuses on quantifying expected losses at each storey and combining them into a system-level vulnerability representation.

To derive system-level vulnerabilities via SLFs, the following logic is implemented:
- For each storey and seismic demand level, calculate the expected storey losses by interpolating from the provided storey loss functions.
- Sum the interpolated storey losses to compute the total building loss at each Intensity Measure (IM) level.
- Fit a regression model to relate the total building loss to IM level.
- Adjust repair-related (non-collapse) losses by the system-level collapse fragility to produce the final vulnerability model for the building class.
