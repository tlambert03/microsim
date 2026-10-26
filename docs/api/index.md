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
module, so they always match the installed version. Each object is documented
once, in the module or subpackage that defines it; most are also importable
from `microsim.schema`, as above. The main objects, by topic:

<div class="ms-api-index" markdown>

## Simulation

| Object | Description |
| --- | --- |
| [`Simulation`][microsim.schema.simulation.Simulation] | the top-level model |
| [`Simulation.run`][microsim.schema.simulation.Simulation.run] | run every stage and return the digital image |
| [`Simulation.run_batch`][microsim.schema.simulation.Simulation.run_batch] | run once per sample, reusing PSFs and emission rates |
| [`Track`][microsim.schema.simulation.Track] | channels acquired at the same time ([guide](../guide/channels.md#sequential-and-simultaneous-acquisition)) |
| [`ground_truth`][microsim.schema.simulation.Simulation.ground_truth], [`filtered_emission_rates`][microsim.schema.simulation.Simulation.filtered_emission_rates], [`optical_image`][microsim.schema.simulation.Simulation.optical_image], [`digital_image`][microsim.schema.simulation.Simulation.digital_image] | the stages, run one at a time ([guide](../guide/index.md)) |

## Spaces

| Object | Description |
| --- | --- |
| [`ShapeScaleSpace`][microsim.schema.space.ShapeScaleSpace], [`ExtentScaleSpace`][microsim.schema.space.ExtentScaleSpace], [`ShapeExtentSpace`][microsim.schema.space.ShapeExtentSpace] | a space given by two of shape, extent, and scale |
| [`DownscaledSpace`][microsim.schema.space.DownscaledSpace], [`UpscaledSpace`][microsim.schema.space.UpscaledSpace] | a space derived from another one ([guide](../guide/spaces.md)) |

## Sample

| Object | Description |
| --- | --- |
| [`Sample`][microsim.schema.sample.Sample], [`FluorophoreDistribution`][microsim.schema.sample.FluorophoreDistribution] | the labels in the sample, and where each one is |
| [`Fluorophore`][microsim.schema.sample.Fluorophore] | spectra, extinction coefficient, quantum yield, lifetime |
| [`MatsLines`][microsim.schema.sample.MatsLines], [`CosemLabel`][microsim.schema.sample.CosemLabel] | built-in distributions ([guide](../guide/samples.md)) |
| [`cosem`][microsim.cosem], [`allen`][microsim.allen] | OpenOrganelle (COSEM) datasets, Allen Institute neuron reconstructions |

## Optics

| Object | Description |
| --- | --- |
| [`ObjectiveLens`][microsim.schema.lens.ObjectiveLens] | numerical aperture, immersion and specimen refractive index |
| [`Widefield`][microsim.schema.modality.Widefield], [`Confocal`][microsim.schema.modality.Confocal], [`Identity`][microsim.schema.modality.Identity] | imaging modalities ([guide](../guide/modalities.md)) |
| [`psf`][microsim.psf] | functions to compute widefield and confocal PSFs directly |

## Channels

| Object | Description |
| --- | --- |
| [`OpticalConfig`][microsim.schema.optical_config.OpticalConfig] | a channel: light sources, filters, and optionally its own exposure (and detector QE) ([guide](../guide/channels.md)) |
| [`LightSource`][microsim.schema.optical_config.LightSource] | an excitation spectrum and its relative power |
| [`Bandpass`][microsim.schema.optical_config.Bandpass], [`Longpass`][microsim.schema.optical_config.Longpass], [`Shortpass`][microsim.schema.optical_config.Shortpass], [`SpectrumFilter`][microsim.schema.optical_config.SpectrumFilter] | filters |
| [`Spectrum`][microsim.schema.spectrum.Spectrum] | a sampled spectrum |
| [`optical_config.lib`][microsim.schema.optical_config.lib] | built-in filter sets and laser lines |

## Detectors

| Object | Description |
| --- | --- |
| [`CameraCCD`][microsim.schema.detectors.CameraCCD], [`CameraCMOS`][microsim.schema.detectors.CameraCMOS], [`CameraEMCCD`][microsim.schema.detectors.CameraEMCCD] | cameras ([guide](../guide/detectors.md)) |
| [`PMT`][microsim.schema.detectors.PMT], [`HyD`][microsim.schema.detectors.HyD] | point detectors |
| [`detectors.lib`][microsim.schema.detectors.lib] | built-in detectors |

## Settings and utilities

| Object | Description |
| --- | --- |
| [`Settings`][microsim.schema.settings.Settings] | array backend, precision, random seed, caching ([guide](../guide/settings.md)) |
| [`util`][microsim.util] | plotting (`ortho_plot`, `ndview`, `animate`) and cache helpers |

</div>
