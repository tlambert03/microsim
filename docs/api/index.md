---
icon: lucide/braces
---

# API reference

All the models used to define a simulation can be imported from
`microsim.schema`, which is usually imported as `ms`:

```py
from microsim import schema as ms

ms.Simulation, ms.Confocal, ms.CameraCMOS, ...
```

Every model is a [pydantic](https://docs.pydantic.dev) model. Fields are
validated when the object is created and when they are assigned, and every
field that takes a model also accepts a dictionary with the same keys.
Unknown fields raise an error.

| Page                                  | Contents                                                        |
| ------------------------------------- | --------------------------------------------------------------- |
| [Simulation](simulation.md)           | `Simulation`, the top-level model and its stage methods         |
| [Spaces](space.md)                    | `ShapeScaleSpace`, `DownscaledSpace`, ...                       |
| [Samples](sample.md)                  | `Sample`, `FluorophoreDistribution`, `Fluorophore`, distributions |
| [Optical configs](optical-config.md)  | `OpticalConfig`, filters, `LightSource`, built-in filter sets   |
| [Modalities](modality.md)             | `Widefield`, `Confocal`, `Identity`, `ObjectiveLens`            |
| [Detectors](detectors.md)             | cameras, point detectors, built-in detectors                    |
| [Settings](settings.md)               | `Settings`                                                      |
| [Spectrum](spectrum.md)               | `Spectrum`                                                      |
| [PSF](psf.md)                         | `microsim.psf`: PSF functions                                   |
| [Utilities](util.md)                  | plotting and cache helpers                                      |
