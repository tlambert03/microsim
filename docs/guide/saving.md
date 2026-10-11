---
icon: lucide/save
---

# Saving and loading

## Saving the result

Set `output_path` and [`run()`][microsim.schema.simulation.Simulation.run] writes the digital
image to disk:

```py
sim = ms.Simulation(..., output_path="result.zarr")
sim.run()
```

| Extension         | Format  | Stores the simulation parameters | Requires       |
| ----------------- | ------- | -------------------------------- | -------------- |
| `.zarr`           | Zarr    | yes                              |                |
| `.nc`             | NetCDF  | yes                              |                |
| `.tif`, `.tiff`   | TIFF    | no (pixel data only)             | `io` extra     |

Zarr and NetCDF files keep the dimension names, coordinates (µm), and channel
names, and store the full simulation as JSON in the
`microsim.Simulation` attribute. Open them with xarray:

```py
import xarray as xr

result = xr.open_dataarray("result.zarr", engine="zarr")
sim = ms.Simulation.model_validate_json(result.attrs["microsim.Simulation"])
```

!!! tip "Prefer Zarr"

    Without the `netCDF4` package installed, xarray writes NetCDF files in the
    older NetCDF-3 format, which has no unsigned integer types. Gray values
    above 32,767 (possible with a 16-bit detector) do not survive the round
    trip. Zarr has no such limit.

To save intermediate stages, use xarray directly, for example
`sim.optical_image().to_netcdf("optical.nc")`.

## Saving the simulation

A `Simulation` is a [pydantic](https://docs.pydantic.dev) model, so it can be
written to and read from JSON:

```py
from pathlib import Path

Path("sim.json").write_text(sim.model_dump_json(indent=2))

sim = ms.Simulation.model_validate_json(Path("sim.json").read_text())
```

The JSON is complete: fluorophores loaded from FPbase are saved with their
spectra, so loading does not need network access. It also includes the random
seed, so loading and running the simulation in the same environment reproduces
the result.

You can also write simulations as JSON (or YAML, or anything that loads into
a dictionary) by hand. Only the fields you want to change from the defaults are
needed:

```json title="confocal.json"
{
  "truth_space": {"shape": [128, 512, 512], "scale": [0.02, 0.01, 0.01]},
  "output_space": {"downscale": 8},
  "sample": [
    {"distribution": {"type": "matslines", "density": 0.5, "length": 30}, "fluorophore": "EGFP"}
  ],
  "channels": ["wKqWbg::Widefield Green"],
  "modality": {"type": "confocal", "pinhole_au": 0.5},
  "detector": {"camera_type": "CMOS", "qe": 0.8, "read_noise": 1.5},
  "settings": {"random_seed": 42},
  "output_path": "confocal.zarr"
}
```

!!! note "What cannot be serialized"

    Custom distribution objects (a class with a `render` method that is not a
    pydantic model) and functions used as `concentration` cannot be written to
    JSON. When writing the output file, microsim then stores the simulation
    without the `sample`.

!!! warning "Record the version"

    A saved simulation describes the parameters, not the implementation.
    microsim is changing quickly: a newer version may reject a saved JSON file,
    or produce a different image from it. Keep `microsim.__version__` with
    your saved simulations and results.
