<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)]

# Vulnerability-Toolkit

![logo](https://github.com/GEMScienceTools/vulnerability-toolkit/blob/main/imgs/gem-vulnerability-toolkit.png)

## 🔎 Overview

The **Vulnerability-Toolkit** is an open source library that provides modelling of multi-degree-of-freedom systems and assessment via nonlinear time-history analyses for vulnerability and risk calculations. The **Vulnerability-Toolkit** is developed by the **[GEM](http://www.globalquakemodel.org)** (Global Earthquake Model) Foundation and its collaborators.

DOI: TBD

## 🛠️ Current Features

* MDOF Modelling: Model single- and multi-degree-of-freedom system using low-level information (e.g., number of storeys, first-mode transformation factor, SDoF- or storey-based force-deformation relationships);
* Modal analysis: Estimate periods of vibration and modal shapes;
* Static analysis: Perform gravity, static and cyclic pushover analyses;
* Dynamic analysis: Perform cloud analysis;
* Regression analysis: Perform regression analysis on cloud analysis data to characterise EDP|IM relationship;
* Fragility analysis: Calculate median seismic intensities, associated dispersion (i.e., record-to-record and modelling uncertainties) and the corresponding probabilities of damage based on demand-based thresholds;
* Vulnerability analysis: Perform fragility function convolution with with consequence models  for structural, and building contents and region-specific non-structural storey-loss functions;
* Plotting: Plot analysis outputs such as model overview, cloud analysis results, demand profiles (i.e., peak storey drifts and peak floor acceleration along the height of the model), fragility functions. Additionally, it is possible to animate the MDoF considering a single run;

## 📚 Documentation

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
