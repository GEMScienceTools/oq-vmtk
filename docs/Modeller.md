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

### `compile_model()`
Compiles the multi-degree-of-freedom oscillator model in OpenSeesPy.

### `plot_model(display_info=True)`
Plots the MDOF model structure using Matplotlib.

### `do_gravity_analysis(nG=100, ...)`
Performs a gravity analysis on the MDOF model.

### `do_modal_analysis(num_modes=3, solver='-genBandArpack', doRayleigh=False, pflag=False)`
Computes the modal properties (natural periods and mode shapes) of the structure.

### `do_spo_analysis(ref_disp, disp_scale_factor, push_dir, phi, ...)`
Performs static pushover (SPO) analysis on the model.

### `do_cpo_analysis(ref_disp, mu, numCycles, push_dir, dispIncr, ...)`
Performs cyclic pushover (CPO) analysis.

### `do_nrha_analysis(fnames, dt_gm, sf, t_max, dt_ansys, nrha_outdir, ...)`
Performs nonlinear response history analysis (NRHA) using ground motion records.

## Example

Please consult "example_2" notebook under "demos" for a demonstration of the structural analysis using the modeller class

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
