<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/GEMScienceTools/vulnerability-toolkit">
    <img src="imgs/gem-vulnerability-toolkit.png" alt="Logo" >
  </a>

  <h3 align="center">Vulnerability Toolkit</h3>

  <p align="center">
    This repository contains an open source library that provides modelling of multi-degree-of-freedom systems and assessment via nonlinear time-history analyses for regional vulnerability and risk calculations. The vulnerability toolkit is developed by the Global Earthquake Model (GEM) Foundation and its collaborators.
    <br />
    <a href="https://github.com/GEMScienceTools/vulnerability-toolkit/docs"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/GEMScienceTools/vulnerability-toolkit/demos">View Demos</a>
    ·
    <a href="https://github.com/GEMScienceTools/vulnerability-toolkit/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/GEMScienceTools/vulnerability-toolkit/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

# 🛠️ Features

* Modelling of Multi-Degree-of-Freedom Oscillators in OpenSees: Model single- and multi-degree-of-freedom system using low-level information (e.g., number of storeys, first-mode transformation factor, SDoF- or storey-based force-deformation relationships);
* Linear and Nonlinear Analysis in OpenSees:
  * Modal Analysis: Estimate Periods of Vibration and Modal Shapes;
  * Gravity Analysis
  * Nonlinear Static Analysis: Perform Static and Cyclic Pushover Analyses to Characterise the Lateral Load-Resisting Response of the System;
  * Dynamic Analysis: Perform Nonlinear Time-History Analyses using Selected Ground-Motion Records;
* Fragility Analysis: Calculate Median Seismic Intensities and Total Associated Dispersion (i.e., Record-to-Record Variability and Modelling Uncertainty):
  * Regression Analysis following Cloud Analysis Method to Characterise EDP|IM Relationship and Calculate Exceedance Probabilities of Damage;
* Vulnerability Analysis: Calculate Vulnerability Functions to Estimate Decision-Variables (Economic- and Human-Based) Conditioned on Ground-Shaking Intensity using Consequence Models (Damage-to-Loss Ratios)
* Plotting: Plot Analysis Outputs
  * Model Overview: OpenSees Model (Nodes and Elements)  
  * Cloud Analysis Results and Fitted IM|EDP Relationship
  * Seismic Demand Profiles: Distribution of Peak Storey Drifts and Peak Floor Accelerations
  * Fragility Functions
  * Vulnerability Functions

# 📚 Documentation

TBD

# 🌟 Contributors

Contributors are gratefully acknowledged and listed in CONTRIBUTORS.txt.

<a href="https://github.com/GEMScienceTools/vulnerability-toolkit/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=GEMScienceTools/vulnerability-toolkit" alt="contrib.rocks image" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# © License

This work is licensed under an AGPL v3 license (https://www.gnu.org/licenses/agpl-3.0.en.html)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

# 🤔 Frequently asked questions

### How to contribute?

You can follow the instructions indicated in the [contributing guidelines](./contribute_guidelines.md). (Work-In-Progress)

### Which version am I seeing? How to change the version?

By default, you will see the files in the repository in the `main` branch. Each version of the model that is released can be accessed is marked with a `tag`. By changing the tag version at the top of the repository, you can change see the files for a given version.

Note that the `main` branch could contain the work-in-progress of the next version of the model.

# 📑 References

TBD

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/GEMScienceTools/vulnerability-toolkit.svg?style=for-the-badge
[contributors-url]: https://github.com/GEMScienceTools/vulnerability-toolkit/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/GEMScienceTools/vulnerability-toolkit.svg?style=for-the-badge
[forks-url]: https://github.com/GEMScienceTools/vulnerability-toolkit/network/members
[stars-shield]: https://img.shields.io/github/stars/GEMScienceTools/vulnerability-toolkit.svg?style=for-the-badge
[stars-url]: https://github.com/GEMScienceTools/vulnerability-toolkit/stargazers
[issues-shield]: https://img.shields.io/github/issues/GEMScienceTools/vulnerability-toolkit.svg?style=for-the-badge
[issues-url]: https://github.com/GEMScienceTools/vulnerability-toolkit/issues
[license-shield]: https://img.shields.io/github/license/GEMScienceTools/vulnerability-toolkit.svg?style=for-the-badge
[license-url]: https://github.com/GEMScienceTools/vulnerability-toolkit/blob/master/LICENSE.txt
