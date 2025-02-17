
# IMCalculator

The `IMCalculator` class computes intensity measures from ground-motion records. It provides methods to calculate response spectra, spectral accelerations, velocity and displacement histories, and other seismic intensity measures.

## Installation

You can install `IMCalculator` by cloning the repository:

```bash
git clone https://github.com/yourusername/repository-name.git
```

Then, navigate into the directory:

```bash
cd repository-name
```

## Usage

To use the `IMCalculator`, first import the class:

```python
from im_calculator import IMCalculator
```

### Constructor

```python
IMCalculator(acc, dt, damping=0.05)
```

**Parameters:**
- `acc` : list or np.array  
  The acceleration time series (in g).
- `dt` : float  
  The time step of the accelerogram (in seconds).
- `damping` : float, optional  
  The damping ratio (default is 5%).

### Methods

#### `get_spectrum(periods=np.linspace(1e-5, 4.0, 100), damping_ratio=0.05)`
Compute the response spectrum using the Newmark-beta method.

**Parameters:**
- `periods` : np.array  
  A list of periods to compute spectral response (in seconds).
- `damping_ratio` : float  
  Damping ratio (default is 5%).

**Returns:**
- `periods` : np.array  
  Periods of the response spectrum.
- `sd` : np.array  
  Spectral displacement (in meters).
- `sv` : np.array  
  Spectral velocity (in m/s).
- `sa` : np.array  
  Spectral acceleration (in g).

#### `get_sa(period)`
Get spectral acceleration for a given period.

**Parameters:**
- `period` : float  
  The target period (in seconds).

**Returns:**
- `sa_interp` : float  
  Spectral acceleration (in g) at the given period.

#### `get_saavg(period)`
Compute geometric mean of spectral accelerations over a range of periods.

**Parameters:**
- `period` : float  
  The conditioning period (in seconds).

**Returns:**
- `sa_avg` : float  
  Average spectral acceleration at the given period.

#### `get_saavg_user_defined(periods_list)`
Compute geometric mean of spectral accelerations for user-defined list of periods.

**Parameters:**
- `periods_list` : list or np.array  
  List of user-defined periods (in seconds) for spectral acceleration calculation.

**Returns:**
- `sa_avg` : float  
  Geometric mean of spectral accelerations over user-defined periods.

#### `get_velocity_displacement_history()`
Compute velocity and displacement history with drift correction.

**Returns:**
- `vel` : np.array  
  Velocity time-history (in m/s).
- `disp` : np.array  
  Displacement time-history (in m).

#### `get_amplitude_ims()`
Compute amplitude-based intensity measures.

**Returns:**
- `pga` : float  
  Peak ground acceleration (in g).
- `pgv` : float  
  Peak ground velocity (in m/s).
- `pgd` : float  
  Peak ground displacement (in meters).

#### `get_arias_intensity()`
Compute Arias Intensity.

**Returns:**
- `AI` : float  
  Arias intensity (in m/s).

#### `get_cav()`
Compute Cumulative Absolute Velocity (CAV).

**Returns:**
- `CAV` : float  
  Cumulative absolute velocity (in m/s).

#### `get_significant_duration(start=0.05, end=0.95)`
Compute significant duration (time between 5% and 95% of Arias intensity).

**Parameters:**
- `start` : float, optional  
  Starting percentage of Arias intensity for duration calculation (default is 0.05).
- `end` : float, optional  
  Ending percentage of Arias intensity for duration calculation (default is 0.95).

**Returns:**
- `sig_duration` : float  
  Significant duration (in seconds).

#### `get_duration_ims()`
Compute duration-based intensity measures: Arias Intensity (AI), CAV, and t_595.

**Returns:**
- `ai` : float  
  Arias intensity (in m/s).
- `cav` : float  
  Cumulative absolute velocity (in m/s).
- `t_595` : float  
  5%-95% significant duration (in seconds).

#### `get_FIV3(period, alpha, beta)`
Computes the filtered incremental velocity IM for a ground motion as per the approach by Dávalos and Miranda (2019).

**Parameters:**
- `period` : float  
  The period (in seconds).
- `alpha` : float  
  Period factor.
- `beta` : float  
  Cut-off frequency factor.

**Returns:**
- `FIV3` : float  
  Intensity measure FIV3 (as per Eq. (3) of Dávalos and Miranda (2019)).
- `FIV` : np.array  
  Filtered incremental velocity (as per Eq. (2)).
- `t` : np.array  
  Time series of FIV.
- `ugf` : np.array  
  Filtered acceleration time history.
- `pks` : np.array  
  Three peak values used to compute FIV3.
- `trs` : np.array  
  Three trough values used to compute FIV3.

## Example

Here's an example of how to use the `IMCalculator`:

```python
from im_calculator import IMCalculator
import numpy as np

# Example ground motion data
acceleration = np.random.rand(1000)  # Replace with actual acceleration data
dt = 0.01  # Time step in seconds

# Initialize the calculator
calculator = IMCalculator(acc=acceleration, dt=dt)

# Get response spectrum
periods, sd, sv, sa = calculator.get_spectrum()

# Get spectral acceleration at a specific period
sa_value = calculator.get_sa(0.5)

# Compute geometric mean of spectral accelerations
sa_avg = calculator.get_saavg(1.0)

# Get velocity and displacement history
vel, disp = calculator.get_velocity_displacement_history()

# Get amplitude-based intensity measures
pga, pgv, pgd = calculator.get_amplitude_ims()
```

## Example

Please consult "example_1" notebook under "demos" for a demonstration of the intensity measures calculation and response spectrum derivation using the im_calculator class

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
