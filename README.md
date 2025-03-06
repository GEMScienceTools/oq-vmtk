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

# 👩‍💻🧑‍💻 Installation

Follow these steps to install the required tools and set up the development environment. Note that this procedure implies the installation of the OpenQuake engine dependencies. This procedure was tested on Mac and Linux OS.
It is highly recommended to use a **virtual environment** to install this tool. A virtual environment is an isolated Python environment that allows you to manage dependencies for this project separately from your system’s Python installation. This ensures that the required dependencies for the OpenQuake engine do not interfere with other Python projects or system packages, which could lead to version conflicts.

1. Open a terminal and navigate to the folder where you intend to install the virtual environment using the "cd" command.

  ```bash
   cd <virtual_environment_directory>
  ```

2. Create a virtual environment using the following command:

  ```bash
   python3 -m venv <virtual_environment_name>
  ```

3. Activate the virtual environment:
* On Linux:

  ```bash
   source <virtual_environment_directory>/bin/activate
  ```

* On Windows:

  ```bash
   <virtual_environment_directory>\Scripts
   activate
  ```

4. Enter (while on virtual environment) the preferred directory for "oq-vmtk" using the "cd" command

  ```bash
   cd <preferred_directory>
  ```

5. Clone the "oq-vmtk" repository

 ```bash
 git clone https://github.com/GEMScienceTools/oq-vmtk.git
 ```

6. Complete the development installation by running the following commands depending on your python version {py-version} (e.g., 310, 311 or 312):
* On Linux

  ```bash
  pip install -r <preferred_directory>/requirements-py<py-version>-linux.txt
  pip install -e .
  ```

* On Windows

  ```bash
  pip install -r <preferred_directory>/requirements-py<py-version>-win64.txt
  pip install -e .
  ```

# 📚 Documentation

[WIP]

# 📼 Demos

The repository includes demo scripts that showcase the functionality of the vulnerability-modellers-toolkit (oq-vmtk). You can find them in the demos folder of the repository.

To run a demo, simply navigate to the demos directory and execute the relevant demo script in Jupyter Lab. Jupyter Lab is automatically installed with oq-vmtk.

1. Open a terminal and activate the virtual environment:
* On Linux:

  ```bash
   source <virtual_environment_directory>/bin/activate
  ```

* On Windows:

  ```bash
   <virtual_environment_directory>\Scripts
   activate
  ```

* To deactivate virtual environment:

  ```bash
   deactivate
  ```

2. Open Jupyter Lab from the terminal:

  ```bash
   jupyter-lab
  ```

3. Navigate to the "demos" folder
4. Run the examples

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

You can follow the instructions indicated in the [contributing guidelines](./contribute_guidelines.md)

# 📑 References

[WIP]

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
