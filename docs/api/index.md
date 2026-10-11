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

The pages in this section are generated from the source code, one per public
module, so they always match the installed version. The most useful ones:

| Module                                                       | Contents                                                          |
| ------------------------------------------------------------ | ----------------------------------------------------------------- |
| [`schema.simulation`][microsim.schema.simulation]            | `Simulation`, the top-level model and its stage methods           |
| [`schema.space`][microsim.schema.space]                      | `ShapeScaleSpace`, `DownscaledSpace`, ...                         |
| [`schema.sample`][microsim.schema.sample]                    | `Sample`, `FluorophoreDistribution`, `Fluorophore`, distributions |
| [`schema.optical_config`][microsim.schema.optical_config]    | `OpticalConfig`, filters, `LightSource`                           |
| [`schema.optical_config.lib`][microsim.schema.optical_config.lib] | built-in filter sets                                         |
| [`schema.modality`][microsim.schema.modality]                | `Widefield`, `Confocal`, `Identity`                               |
| [`schema.lens`][microsim.schema.lens]                        | `ObjectiveLens`                                                   |
| [`schema.detectors`][microsim.schema.detectors]              | cameras and point detectors                                       |
| [`schema.detectors.lib`][microsim.schema.detectors.lib]      | built-in detectors                                                |
| [`schema.settings`][microsim.schema.settings]                | `Settings`                                                        |
| [`schema.spectrum`][microsim.schema.spectrum]                | `Spectrum`                                                        |
| [`psf`][microsim.psf]                                        | PSF functions                                                     |
| [`util`][microsim.util]                                      | plotting and cache helpers                                        |
| [`allen`][microsim.allen]                                    | neuron reconstructions from the Allen Institute                   |
| [`cosem`][microsim.cosem]                                    | access to OpenOrganelle (COSEM) datasets                          |

Each object is documented once, in the module or subpackage that defines it.
Most of them are also importable from `microsim.schema`, as above.
