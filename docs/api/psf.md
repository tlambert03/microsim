# PSF

Functions to compute point spread functions. Simulations call these through
their [modality](modality.md); they can also be used on their own.

```python
from microsim.psf import make_psf
from microsim.schema import ObjectiveLens

psf = make_psf(nz=65, nx=65, dx=0.04, dz=0.04, objective=ObjectiveLens(), em_wvl_nm=525)
```

::: microsim.psf
    options:
      show_root_heading: false
      members:
        - make_psf
        - make_confocal_psf
        - vectorial_psf
        - vectorial_psf_centered
