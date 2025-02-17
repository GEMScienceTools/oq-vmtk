# PostProcessor

The `postprocessor` class is designed to process the results of nonlinear time-history analyses. It includes functions for performing cloud analysis, fragility and vulnerability assessments, and calculating annual probabilities of damage and loss.

## Installation

Clone the repository and navigate into the directory:

```bash
git clone https://github.com/yourusername/repository-name.git
cd repository-name
```

## Usage

To use the `postprocessor` class, first import it:

```python
from postprocessor import postprocessor
```

### Constructor

```python
postprocessor()
```

This class does not require initialization parameters.

## Methods

### `do_cloud_analysis(imls, edps, damage_thresholds, lower_limit, censored_limit, sigma_build2build=0.3, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3))`
Performs censored cloud analysis using engineering demand parameters and intensity measures.

**Parameters:**
- `imls`: list  
  Intensity measure levels.
- `edps`: list  
  Engineering demand parameters.
- `damage_thresholds`: list  
  Damage thresholds for different damage states.
- `lower_limit`: float  
  Minimum EDP threshold for filtering records.
- `censored_limit`: float  
  Maximum EDP threshold for filtering records.
- `sigma_build2build`: float  
  Modelling uncertainty (default: 0.3).
- `intensities`: np.array  
  Range of intensity measures for fragility calculation.

**Returns:**
- `cloud_dict`: dict  
  Contains regression coefficients, fragility parameters, and probabilities of exceedance.

### `get_fragility_function(theta, beta_total, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3))`
Computes a lognormal cumulative distribution function for damage state exceedance.

**Parameters:**
- `theta`: float  
  Median seismic intensity.
- `beta_total`: float  
  Total uncertainty.
- `intensities`: np.array  
  Range of intensity measure levels.

**Returns:**
- `poes`: list  
  Probabilities of damage exceedance.

### `get_vulnerability_function(poes, consequence_model, intensities=np.round(np.geomspace(0.05, 10.0, 50), 3))`
Computes vulnerability functions based on fragility and consequence models.

**Parameters:**
- `poes`: np.array  
  Probabilities of exceedance per damage state.
- `consequence_model`: list  
  Damage-to-loss ratios.
- `intensities`: np.array  
  Range of intensity measure levels.

**Returns:**
- `loss`: np.array  
  Expected loss ratios.

### `calculate_sigma_loss(loss)`
Computes the uncertainty in loss estimates.

**Parameters:**
- `loss`: list  
  Expected loss ratios.

**Returns:**
- `sigma_loss_ratio`: list  
  Standard deviation of loss estimates.
- `a_beta_dist`: float  
  Beta distribution coefficient.
- `b_beta_dist`: float  
  Beta distribution coefficient.

### `calculate_average_annual_damage_probability(fragility_array, hazard_array, return_period=1, max_return_period=5000)`
Computes the average annual probability of exceeding a damage state.

**Parameters:**
- `fragility_array`: np.array  
  Intensity measure levels and probabilities of exceedance.
- `hazard_array`: np.array  
  Intensity measure levels and annual rates of exceedance.
- `return_period`: int  
  Return period of interest (default: 1 year).
- `max_return_period`: int  
  Maximum return period considered (default: 5000 years).

**Returns:**
- `average_annual_damage_probability`: float  
  Expected annual probability of exceeding a damage state.

### `calculate_average_annual_loss(vulnerability_array, hazard_array, return_period=1, max_return_period=5000)`
Computes the average annual loss probability based on vulnerability and hazard models.

**Parameters:**
- `vulnerability_array`: np.array  
  Intensity measure levels and associated loss ratios.
- `hazard_array`: np.array  
  Intensity measure levels and annual rates of exceedance.
- `return_period`: int  
  Return period of interest (default: 1 year).
- `max_return_period`: int  
  Maximum return period considered (default: 5000 years).

**Returns:**
- `average_annual_loss`: float  
  Expected annual loss probability.

## Example

Please consult [example_3](https://github.com/GEMScienceTools/vulnerability-toolkit/blob/main/demos/example_3.ipynb) notebook under the [demos](https://github.com/GEMScienceTools/vulnerability-toolkit/blob/main/demos/) folder for a demonstration of postprocessing cloud analysis results using the `postprocessor` class.
