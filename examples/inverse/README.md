# Differentiable inverse problems

microsim's forward model is written in JAX (the PSF is built from a Zernike
pupil via [chromatix](https://github.com/chromatix-team/chromatix), then
convolved with the sample), so it is differentiable end-to-end.  That lets you
phrase classic microscopy problems as gradient-descent optimizations with
[optax](https://optax.readthedocs.io).

Shared helpers (loss functions, an optax fit loop, non-negativity
reparameterization) live in [`microsim.inverse`](../../src/microsim/inverse.py);
the PSF model is [`microsim.optics.vectorial_psf`](../../src/microsim/optics).

| Example | Optimization variable | What it recovers |
|---|---|---|
| [`phase_retrieval.py`](phase_retrieval.py) | Zernike coefficients | the pupil aberration, from a measured through-focus PSF stack |
| [`deconvolution.py`](deconvolution.py) | sample density (per voxel) | the fluorophore distribution, from a blurred + photon-noisy image |
| [`adaptive_optics.py`](adaptive_optics.py) | corrective Zernike coefficients | a deformable-mirror correction that maximizes focal sharpness (sensorless AO) |

```bash
python examples/inverse/phase_retrieval.py
python examples/inverse/deconvolution.py
python examples/inverse/adaptive_optics.py
```

Each script writes a `*.png` summary figure and prints quantitative results.
They run on CPU in a minute or two; on a GPU they are far faster.
