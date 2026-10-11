---
icon: lucide/settings
---

# Settings

[`Settings`][microsim.schema.Settings] holds options that affect how a
simulation is computed, but not what it represents.

```py
sim = ms.Simulation(..., settings=ms.Settings(random_seed=42, np_backend="jax"))
```

| Setting                              | Default   | Effect                                                         |
| ------------------------------------ | --------- | -------------------------------------------------------------- |
| `random_seed`                        | random    | seed for the ground truth and noise                            |
| `np_backend`                         | `"auto"`  | array library used for computation                             |
| `float_dtype`                        | `float32` | floating-point precision                                       |
| `max_psf_radius_aus`                 | 6         | lateral size limit of each PSF, in Airy radii                  |
| `spectral_bins_per_emission_channel` | 1         | number of PSFs used per fluorophore and channel                |
| `spectral_bin_threshold_percentage`  | 1         | part of the emission spectrum that is ignored when binning     |
| `cache`                              | on        | read and write the on-disk ground-truth cache                  |

## Reproducibility

Random numbers are used to generate some ground truths (such as
[`MatsLines`][microsim.schema.MatsLines]) and to draw every kind of noise.
When the `Simulation` is created without a `random_seed`, a random one is
chosen. It is stored on the object, so `sim.settings.random_seed` tells you
which seed was used, and saving the simulation to JSON records it.

With a fixed seed, the same simulation run twice in the same environment gives
the same result. Different backends use different random number generators,
so results are not identical across backends.

!!! warning "Reproducing results across versions"

    microsim is under active development, and the same parameters and seed may
    give different results in a different version of microsim, as models are
    corrected and extended. Record the version (`microsim.__version__`) with
    your results, and pin it in your environment.

## Array backends

Computation is done with NumPy by default. Most of the time goes into PSF
calculation and FFT convolution, which can run on a GPU:

| `np_backend` | Library                                     |
| ------------ | ------------------------------------------- |
| `"numpy"`    | NumPy and SciPy (CPU)                       |
| `"cupy"`     | [CuPy](https://cupy.dev) (NVIDIA GPU)       |
| `"jax"`      | [JAX](https://docs.jax.dev) (CPU or GPU)    |
| `"torch"`    | PyTorch (incomplete)                        |
| `"auto"`     | CuPy if it can be imported, otherwise NumPy |

The backend can also be chosen with the `MICROSIM_BACKEND` environment
variable, which is used when `np_backend` is `"auto"`.

With a GPU backend, the arrays inside the returned `DataArray` live on the
GPU. Use `.values` or `numpy.asarray()` to get a NumPy array.

## Accuracy vs. speed

**`max_psf_radius_aus`** limits the lateral size of each PSF to this many Airy
radii from the center. A smaller value makes convolution much faster but
drops the faint outer parts of the PSF, which in widefield carry a noticeable
fraction of the out-of-focus light. Set it to `None` to use PSFs as large as
the truth space.

**`spectral_bins_per_emission_channel`** sets how many wavelengths are used to
represent each fluorophore's detected emission. With 1 (the default), one PSF
at the centroid wavelength is used. With more, the emission spectrum (after
dropping the parts below `spectral_bin_threshold_percentage` of its peak) is
split into that many bins, and one PSF is computed per bin and weighted by
its photons. Time and memory grow with the number of bins.

**`float_dtype`** can be set to `float64` for higher precision, at twice the
memory.

## Caching

microsim caches results that are slow to compute and depend only on their
parameters:

- **PSFs**, in memory (the most recent 64) and on disk.
- **Fourier transforms of PSFs**, in memory, up to
  `Settings.cache.in_mem_size.otf_mb` (1000 MB by default; GPU memory for GPU
  backends).
- **Emission rates** (the result of `filtered_emission_rates`), in memory, for
  the most recent 32 combinations of channels, fluorophores, detector QE, and
  objective.
- **Ground truth** from `MatsLines` and `CosemLabel`, on disk, keyed by the
  distribution's parameters, the truth space, the concentration, and the random
  seed.
- **OpenOrganelle downloads**, on disk.

The disk cache is in the user cache directory given by
[platformdirs](https://github.com/tox-dev/platformdirs) (for example
`~/Library/Caches/microsim` on macOS, `~/.cache/microsim` on Linux):

```py
from microsim.util import microsim_cache, clear_cache

microsim_cache()  # path to the cache directory
clear_cache()  # delete everything
clear_cache("psf/*")  # delete matching entries
```

To turn caching off:

- `Settings(cache=False)` disables reading and writing the ground-truth cache.
  `Settings(cache={"read": True, "write": False})` controls the two
  separately.
- The environment variable `MICROSIM_CACHE=0` disables both the ground-truth
  and the PSF disk caches.

## Environment variables

Every setting can be set with an environment variable named `MICROSIM_`
followed by the setting name. Nested settings use `__` as a separator:

```bash
export MICROSIM_RANDOM_SEED=42
export MICROSIM_NP_BACKEND=jax
export MICROSIM_CACHE__WRITE=0   # read the cache, but don't write to it
```

Values passed in code take precedence over environment variables.
