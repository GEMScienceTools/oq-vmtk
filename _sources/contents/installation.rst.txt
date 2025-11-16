Installation
############

Follow these steps to install the oq-vmtk package and its dependencies. Note that
this procedure implies the installation of the OpenQuake engine dependencies.
This procedure was tested on Windows and Linux OS. It is highly recommended
to use a virtual environment to install this tool. A virtual environment is an
isolated Python environment that allows you to manage dependencies for this
project separately from your system’s Python installation. This ensures that the
required dependencies for the OpenQuake engine do not interfere with other Python
projects or system packages, which could lead to version conflicts.

### 1. Clone the Repository

Open your terminal, and run:

.. code-block:: bash

   cd <virtual_environment_directory>
   git clone https://github.com/GEMScienceTools/oq-vmtk.git
   cd oq-vmtk

### 2. Set Up a Virtual Environment (Recommended)

Create a virtual environment to manage dependencies:

.. code-block:: bash

   python -m venv .venv  # On Windows
   python3 -m venv .venv  # On Linux

Activate the virtual environment:

.. code-block:: bash

   .venv\Scripts\activate  # On Windows
   source .venv/Scripts/activate  # On Linux

### 3. Install Dependencies

Install the required packages listed in `requirements.txt`.

**For Windows Users:** Install the appropriate requirements file based on your Python version:

.. code-block:: bash

   pip install -r requirements-py310-win64.txt  # Python 3.10
   pip install -r requirements-py311-win64.txt  # Python 3.11
   pip install -r requirements-py312-win64.txt  # Python 3.12

**For Linux Users:** Install the appropriate requirements file based on your Python version:

.. code-block:: bash

   pip install -r requirements-py310-linux64.txt  # Python 3.10
   pip install -r requirements-py311-linux64.txt  # Python 3.11
   pip install -r requirements-py312-linux64.txt  # Python 3.12

**Note:** To check your current Python version, run the following command:

.. code-block:: bash

   python --version

### 4. Install the Package

Install the `oq-vmtk` package in editable mode:

.. code-block:: bash

   pip install -e .


.. note::

   **For macOS Users:** OpenSeesPy does not currently support macOS versions running
   on arm64 processors, such as M1 and M2 chips. As a result, newer OpenSeesPy versions
   are not available for macOS. To use OpenSeesPy on a Mac, it is advised to run a virtual
   machine with Linux or Windows.
