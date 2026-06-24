# Development Notes

Setup friction encountered on this fork during GPU development — kept
here separately so the main README stays clean. Not universal microsim
guidance, just what was hit on this machine.

## GPU backend (cupy) setup

1. **numpy version conflict** — microsim's metadata pins `numpy<2.0`, but
   `cupy-cuda12x` requires `numpy>=2.0`. In practice, numpy 2.4.6 worked
   fine at runtime with this version of microsim despite the metadata
   warning — if `pip` flags the conflict, it's likely safe to ignore,
   but verify by running a real simulation before trusting it.

2. **CUDA headers not found** (`RuntimeError: Failed to find CUDA headers`)
   — if your system has multiple CUDA toolkit installs (common after
   driver updates over time), cupy's JIT compiler may not find headers
   even with `CUDA_PATH` set correctly. Installing the toolkit extra
   directly into the venv sidesteps the problem:
```bash
   pip install "cupy-cuda12x[ctk]"
```

3. **zarr / numcodecs mismatch** (`ImportError: cannot import name
   'cbuffer_sizes' from 'numcodecs.blosc'`) — zarr 2.17.x expects an
   older numcodecs API:
```bash
   pip install "numcodecs<0.13"
```

4. **COSEM data fetching** — loading real segmentation data (e.g. the
   `cosem` example) requires the optional extra:
```bash
   pip install "microsim[cosem]"
```

None of these are bugs in microsim itself, except where already filed
as upstream PRs (#140, #141) — mostly environment/dependency friction
from multiple existing CUDA/Python installs on a dev machine.
