# Modeller

The `modeller` class is designed to model and analyze multi-degree-of-freedom (MDOF) oscillators using OpenSeesPy. It provides functionalities for model generation, static and cyclic pushover analysis, modal analysis, and nonlinear time history analysis.

## Installation

Clone the repository and navigate into the directory:

```bash
git clone https://github.com/yourusername/repository-name.git
cd repository-name
```

## Usage

To use the `modeller` class, first import it:

```python
from modeller import modeller
```

### Constructor

```python
modeller(number_storeys, floor_heights, floor_masses, storey_disps, storey_forces, degradation)
```

**Parameters:**
- `number_storeys` : int  
  Number of storeys in the MDOF system.
- `floor_heights` : list  
  List of floor heights in meters.
- `floor_masses` : list  
  List of floor masses in tonnes.
- `storey_disps` : np.array  
  Storey displacements.
- `storey_forces` : np.array  
  Storey forces.
- `degradation` : bool  
  Flag to enable/disable hysteretic degradation.

## Methods

### `create_Pinching4_material(mat1Tag, mat2Tag, storey_forces, storey_disps, degradation)`
Creates a `Pinching4` material model for nonlinear elements.

**Parameters:**
- `mat1Tag`: int  
  Material tag for primary material.
- `mat2Tag`: int  
  Material tag for secondary material.
- `storey_forces`: np.array  
  Array of storey force values.
- `storey_disps`: np.array  
  Array of storey displacement values.
- `degradation`: bool  
  Flag to enable/disable degradation effects.

### `compile_model()`
Compiles the multi-degree-of-freedom oscillator model in OpenSeesPy.

### `plot_model(display_info=True)`
Plots the MDOF model structure using Matplotlib.

**Parameters:**
- `display_info`: bool  
  Whether to display node information in the plot.

### `do_gravity_analysis(nG=100, ansys_soe='BandGeneral', constraints_handler='Transformation', numberer='RCM', test_type='NormDispIncr', init_tol=1.0e-6, init_iter=50, algorithm_type='Newton')`
Performs a gravity analysis on the MDOF model.

**Parameters:**
- `nG`: int  
  Number of gravity steps.
- `ansys_soe`: str  
  Type of system of equations solver.
- `constraints_handler`: str  
  Type of constraints handler used.
- `numberer`: str  
  Numbering scheme for degrees of freedom.
- `test_type`: str  
  Type of convergence test.
- `init_tol`: float  
  Initial tolerance for convergence.
- `init_iter`: int  
  Maximum number of iterations.
- `algorithm_type`: str  
  Solution algorithm.

### `do_modal_analysis(num_modes=3, solver='-genBandArpack', doRayleigh=False, pflag=False)`
Computes the modal properties (natural periods and mode shapes) of the structure.

**Parameters:**
- `num_modes`: int  
  Number of vibration modes to compute.
- `solver`: str  
  Type of solver used.
- `doRayleigh`: bool  
  Whether to include Rayleigh damping.
- `pflag`: bool  
  Whether to print analysis results.

### `do_spo_analysis(ref_disp, disp_scale_factor, push_dir, phi, num_steps=200)`
Performs static pushover (SPO) analysis on the model.

**Parameters:**
- `ref_disp`: float  
  Reference displacement.
- `disp_scale_factor`: float  
  Scale factor for displacement.
- `push_dir`: int  
  Direction of pushover (1 = X, 2 = Y, 3 = Z).
- `phi`: list  
  Shape of lateral load applied.
- `num_steps`: int  
  Number of pushover analysis steps.

### `do_cpo_analysis(ref_disp, mu, numCycles, push_dir, dispIncr)`
Performs cyclic pushover (CPO) analysis.

**Parameters:**
- `ref_disp`: float  
  Reference displacement.
- `mu`: float  
  Target ductility.
- `numCycles`: int  
  Number of loading cycles.
- `push_dir`: int  
  Direction of loading.
- `dispIncr`: float  
  Displacement increment.

### `do_nrha_analysis(fnames, dt_gm, sf, t_max, dt_ansys, nrha_outdir, xi=0.05)`
Performs nonlinear response history analysis (NRHA) using ground motion records.

**Parameters:**
- `fnames`: list  
  Filepaths of ground motion records.
- `dt_gm`: float  
  Time step of the ground motion.
- `sf`: float  
  Scale factor for the records.
- `t_max`: float  
  Maximum duration of analysis.
- `dt_ansys`: float  
  Analysis time step.
- `nrha_outdir`: str  
  Output directory for NRHA results.
- `xi`: float  
  Damping ratio (default 5%).

## Example

Please consult [example_2](https://github.com/GEMScienceTools/vulnerability-toolkit/blob/main/demos/example_2.ipynb) notebook under the [demos](https://github.com/GEMScienceTools/vulnerability-toolkit/blob/main/demos/) folder for a demonstration of the structural analysis using the modeller class
