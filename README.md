# microsim

[![License](https://img.shields.io/pypi/l/microsim.svg?color=green)](https://github.com/tlambert03/microsim/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/microsim.svg?color=green)](https://pypi.org/project/microsim)
[![Python Version](https://img.shields.io/pypi/pyversions/microsim.svg?color=green)](https://python.org)
[![CI](https://github.com/tlambert03/microsim/actions/workflows/ci.yml/badge.svg)](https://github.com/tlambert03/microsim/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tlambert03/microsim/branch/main/graph/badge.svg)](https://codecov.io/gh/tlambert03/microsim)
[![DOI](https://zenodo.org/badge/483645397.svg)](https://doi.org/10.5281/zenodo.18942661)

Light microscopy simulation in python.

microsim is **JAX-based and differentiable**: you describe a microscope with
declarative (pydantic) models and get a simulated image out, but because the
whole forward model — including a vectorial, Zernike-pupil PSF — is written in
JAX, you can also differentiate through it and solve physics-based optimization
problems with gradient descent (phase retrieval, deconvolution, adaptive optics;
see [`examples/inverse/`](examples/inverse)).

The goal of this library is to generate highly realistic simulated data such as the following:

![Montage2](https://github.com/tlambert03/microsim/assets/1609449/4bc9eb85-b275-4315-b60d-2cb3d003b7f1)

*(this data was generated using segmentations from
[cell jrc_hela-3 at OpenOrganelle](https://openorganelle.janelia.org/datasets/jrc_hela-3) as the ground truth)*

<https://github.com/user-attachments/assets/905fa1af-c8bd-406e-87a7-7c5921de74d1>

## Documentation

Start with the [tutorial](https://www.talleylambert.com/microsim/tutorial/), and read about the
[simulation stages](https://www.talleylambert.com/microsim/stages/), then see the 
[API Reference](<https://tlambert03.github.io/microsim/api/>) for details
on the `Simulation` object and options for all of the fields.

## Installation

### from PyPI

```bash
pip install "microsim[all]"
```

> [!NOTE]
> At the moment, this library is a bit more "application" than it
> is "library".  If you are following the docs or tutorials, its
> probably best to install the full `[all]` extra, which brings in
> io, visualization, and other data-fetching dependencies.  However,
> the bare minimal install is fully functional for generating data.

### Local Development

```sh
gh repo clone tlambert03/microsim
cd microsim
uv sync
```

### From github

To get the bleeding edge version, which will change rapidly, you can install from github.

```bash
pip install "microsim[all] @ git+https://github.com/tlambert03/microsim"
```

> [!NOTE]
> microsim is JAX-only and requires Python ≥ 3.12. The PSF model is backed by
> [chromatix](https://github.com/chromatix-team/chromatix); until its modern
> release is on PyPI, that dependency is pinned to a GitHub commit (see
> `pyproject.toml`).

### With GPU support

microsim runs on the CPU by default. For GPU acceleration, install the
appropriate `jax` wheel for your platform following the
[jax installation instructions](https://jax.readthedocs.io/en/latest/installation.html);
the rest of microsim will then run on the GPU automatically.

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

### Aberrations (Zernike pupil)

The PSF is built from a complex pupil, so you can add arbitrary aberrations as
Zernike coefficients (wavefront OPD in microns, ANSI/OSA indexing) on the
objective lens:

```python
from microsim import schema as ms

objective = ms.ObjectiveLens(
    numerical_aperture=1.4,
    aberration=ms.ZernikeAberration(
        ansi_indices=(5, 12),      # vertical astigmatism, primary spherical
        coefficients=(0.05, 0.03), # microns RMS wavefront
    ),
)
```

### Differentiable optimization

Because the forward model is JAX, you can take gradients with respect to its
parameters and solve inverse problems with [optax](https://optax.readthedocs.io).
`microsim.optics.vectorial_psf` builds a differentiable PSF and
`microsim.inverse` has small helpers (losses, an Adam fit loop). For example,
recovering a pupil aberration from a measured PSF z-stack:

```python
from microsim.optics import vectorial_psf
from microsim.inverse import fit, normalized_mse, zernike_coeffs

ANSI = (5, 6, 7, 8, 9, 12)
cfg = dict(nz=11, nx=48, dxy=0.05, dz=0.15, wvl=0.55, na=1.3, ni=1.515)
measured = vectorial_psf(**cfg, ansi_indices=ANSI, coefficients=[0.05, -0.03, 0.04, 0.02, 0, 0.04])

def loss(c):
    return normalized_mse(vectorial_psf(**cfg, ansi_indices=ANSI, coefficients=c), measured)

result = fit(loss, zernike_coeffs(ANSI), steps=150)  # -> result.params recovers the coefficients
```

See [`examples/inverse/`](examples/inverse) for complete, runnable scripts:
**phase retrieval**, **deconvolution / sample estimation**, and
**sensorless adaptive optics**.
