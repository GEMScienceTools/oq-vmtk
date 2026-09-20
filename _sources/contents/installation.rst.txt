Installation
############

Follow these steps to install the ``oq-vmtk`` package and its dependencies. Note that
this procedure implies the installation of the OpenQuake engine dependencies.
This procedure was tested on Windows and Linux OS. It is highly recommended
to use a virtual environment to install this tool. A virtual environment is an
isolated Python environment that allows you to manage dependencies for this
project separately from your system's Python installation. This ensures that the
required dependencies for the OpenQuake engine do not interfere with other Python
projects or system packages, which could lead to version conflicts.

Clone the Repository
--------------------

Open your terminal, and run:

.. code-block:: bash

   cd <virtual_environment_directory>
   git clone https://github.com/GEMScienceTools/oq-vmtk.git
   cd oq-vmtk

Set Up a Virtual Environment
----------------------------

Create a virtual environment to manage dependencies:

.. code-block:: bash

   python -m venv .venv   # On Windows
   python3 -m venv .venv  # On Linux / macOS

Activate the virtual environment:

.. code-block:: bash

   .venv\Scripts\activate        # On Windows
   source .venv/bin/activate     # On macOS
   source .venv/Scripts/activate # On Linux

Install Dependencies
--------------------

Install the appropriate requirements file based on your operating system and
Python version. Pinned-dependency files are provided for Python 3.11, 3.12 and
3.13 on Linux, Windows, and macOS (arm64).

To check your current Python version, run:

.. code-block:: bash

   python --version

Then pick the matching file. The naming convention is
``requirements-py<MAJOR><MINOR>-<os>.txt`` (for example
``requirements-py312-linux.txt`` for Python 3.12 on Linux):

.. code-block:: bash

   # Linux (replace 312 with 311 or 313 to match your Python version)
   pip install -r requirements-py312-linux.txt

   # Windows
   pip install -r requirements-py312-win64.txt

   # macOS (arm64)
   pip install -r requirements-py312-macos_arm64.txt

Install the Package
-------------------

**Standard install (recommended for users):**

.. code-block:: bash

   pip install .

**Editable / developer install** — use this only if you plan to modify the
source:

.. code-block:: bash

   pip install -e .

Verify the Installation
-----------------------

.. code-block:: bash

   python -c "import openquake.vmtk; print(openquake.vmtk.__version__)"

This should print the installed version (e.g., ``1.0.0``). The version reported
here is the same version archived on Zenodo (DOI
`10.5281/zenodo.17524871 <https://doi.org/10.5281/zenodo.17524871>`_).
