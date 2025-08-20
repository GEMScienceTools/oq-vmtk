# Model Compilation Demo

This demonstration presents a structured workflow for the calibration and development of simplified structural models in OpenSees for regional vulnerability/risk applications, with a focus on building models that represent typical structures yet in an idealised manner such as simplified single-degree-of-freedom (SDOF) and more realistic multi-degree-of-freedom (MDOF) stick-and-mass models with a calibration procedure linking the two approaches.

This Jupyter Notebook provides an example application of the `modeller` and `calibration` modules for creating idealised structural representations of the regional building stock.

The first example includes construction of SDOF representations by defining essential structural properties such as mass, height, and nonlinear capacity.

The second example includes the development of MDOF stick-and-mass models, in which each floor is represented as a lumped mass connected by nonlinear springs capturing lateral stiffness, yielding and nonlinear behaviour.

The third example considers the calibration of MDOF Models Based on SDOF capacity and parameters. Establishing consistency between the SDOF and MDOF representations is carried out by calibrating inter-story properties of the MDOF model using target capacity curves derived from the SDOF model.
