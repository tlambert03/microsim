# microsim

[![License](https://img.shields.io/pypi/l/microsim.svg?color=green)](https://github.com/tlambert03/microsim/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/microsim.svg?color=green)](https://pypi.org/project/microsim)
[![Python Version](https://img.shields.io/pypi/pyversions/microsim.svg?color=green)](https://python.org)
[![CI](https://github.com/tlambert03/microsim/actions/workflows/ci.yml/badge.svg)](https://github.com/tlambert03/microsim/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tlambert03/microsim/branch/main/graph/badge.svg)](https://codecov.io/gh/tlambert03/microsim)

Light microscopy simulation in python

## Installation

### from PyPI

```bash
pip install microsim
```

### from github

To get the bleeding edge version, which will change rapidly, you can install from github.

```bash
pip install git+https://github.com/tlambert03/microsim
```

If available, microsim can use either Jax or Cupy to accelerate computations.
These are not installed by default, see the
[jax](https://jax.readthedocs.io/en/latest/installation.html)
or [cupy](https://docs.cupy.dev/en/stable/install.html) installation instructions,
paying attention to your GPU requirements.  Support for torch is planned.

## Usage

Construct and run a
[`microsim.Simulation`](https://www.talleylambert.com/microsim/api/#microsim.schema.simulation.Simulation)
object.

```python
from microsim import schema as ms
from microsim.util import ortho_plot

# define the parameters of the simulation
sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(shape=(128, 512, 512), scale=(0.02, 0.01, 0.01)),
    output_space={'downscale': 8},
    sample=ms.Sample(
        labels=[ms.MatsLines(density=0.5, length=30, azimuth=5, max_r=1)]
    ),
    modality=ms.Confocal(pinhole_au=0.2),
    output_path="au02.tiff",
)

# run it
result = sim.run()

# optionally plot the result
ortho_plot(result)
```

## Documentation

See the API Reference (<https://tlambert03.github.io/microsim/api/>) for details
on the `Simulation` object and options for all of the fields.
