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

  <h3 align="center">Vulnerability Modeller's ToolKit (OQ-VMTK)</h3>

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


# ✨ Key Features

## 🏗️ Single/Multi-Degree-of-Freedom System Modeling
- Effortlessly create single- and multi-degree-of-freedom models using intuitive low-level parameters.
- Define structures with key attributes like storey count, first-mode transformation factors, and force-deformation relationships.

## 🔍 Comprehensive Analysis Suite
### 📊 Linear & Nonlinear Analysis
- **Modal Analysis:** Extract vibration periods and mode shapes with precision.
- **Gravity Analysis:** Ensure stability before running advanced simulations.
- **Nonlinear Static Analysis:** Perform static and cyclic pushover tests to assess lateral load resistance.
- **Dynamic Time-History Analysis:** Simulate real-world earthquake scenarios using selected ground-motion records.

### 🌍 Seismic Fragility & Vulnerability Assessment
- **Fragility Analysis:** Quantify seismic performance by computing median seismic intensities and total dispersion (record-to-record variability & modeling uncertainty).
- **Regression Analysis:** Characterize EDP|IM relationships using Cloud Analysis and determine damage exceedance probabilities.
- **Vulnerability Analysis:** Derive vulnerability functions to estimate economic and human-based decision variables, incorporating damage-to-loss ratios.

### 📈 Powerful Visualization & Plotting Tools
- Generate insightful plots for:
  - **Model Overview:** Nodes and elements in your OpenSees model.
  - **Cloud Analysis Results:** Fitted IM|EDP relationships.
  - **Seismic Demand Profiles:** Peak storey drifts and floor accelerations.
  - **Fragility Functions:** Visualize probability-based structural performance.
  - **Vulnerability Functions:** Understand risk and loss estimates.

# 🚀 Get Started

## 👩‍💻🧑‍💻 Installation

Follow these steps to install the required tools and set up the development environment. Note that this procedure implies the installation of the OpenQuake engine dependencies. This procedure was tested on Windows and Linux OS.
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
   <virtual_environment_directory>\Scripts\Activate.ps1
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
  pip install -r requirements-py-linux.txt
  pip install -e .
  ```

* On Windows

  ```bash
  pip install -r requirements-py-win64.txt
  pip install -e .
  ```

## 📼 Demos

The repository includes demo scripts that showcase the functionality of the vulnerability-modellers-toolkit (oq-vmtk). You can find them in the demos folder of the repository.

To run a demo, simply navigate to the demos directory and execute the relevant demo script in Jupyter Lab. Jupyter Lab is automatically installed with oq-vmtk.

1. Open a terminal and activate the virtual environment:
* On Linux:

  ```bash
   source <virtual_environment_directory>/bin/activate
  ```

* On Windows:

  ```bash
   <virtual_environment_directory>\Scripts\Activate.ps1
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

# 📚 Documentation

[WIP]

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
