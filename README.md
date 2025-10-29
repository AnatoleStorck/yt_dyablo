A yt frontend for Dyablo, packaged as an extension for yt.


# Installation

```bash
python -m pip install yt_dyablo
```

# Usage

```python
import yt
from yt_dyablo.api import DyabloDataset

ds = DyabloDataset("path_to_dyablo_dataset.hdf5")
```

The loader accepts several parameters
- `field_mapping`: A dictionary to map Dyablo field names (e.g. `rho`) to yt field names (e.g. `density`) with units (e.g. `g/cm**3` or `code_density`).
- `length_unit`, `mass_unit`, `time_unit`, `magnetic_unit`, `velocity_unit`: The units to use for length, mass, time, magnetic field, and velocity respectively.
- `gamma0`: The adiabatic index of the gas.
- `mean_molecular_weight`: The mean molecular weight of the gas (to convert between density/pressure and temperature). For the moment, it is assumed to be constant throughout the dataset.
