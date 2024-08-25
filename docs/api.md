# API Reference

All classes here are importable from the `microsim.schema` module.

## Simulation

The top level object is the `Simulation` object. This
object contains all the information needed to run a simulation.

::: microsim.schema.simulation

## Spaces

Spaces define the shape and scale of various spaces in the simulation.
They can be defined in many different ways, but they all either directly
declare or calculate the `shape`, `scale`, `extent` and `axes` names of
the space in which the simulation is run.

For example, you might define the ground truth fluorophore distribution
in a highly sampled space, and then downscale the resulting simulation
to a realistic microscope image sampling.

```python
from microsim import Simulation
import microsim.schema as ms

Simulation(
    # ground truth with z-step 20nm and XY pixel size 10nm
    truth_space=ms.ShapeScaleSpace(shape=(256, 1024, 1024), scale=(0.02, 0.01, 0.01)),
    # downscale output to 160nm x 80nm x 80nm pixels with shape (32, 128, 128)
    output_space=ms.DownscaledSpace(downscale=8),
    sample=ms.Sample(labels=[])  # ...
)
```

If one of the spaces is defined using a relative space such as `DownscaledSpace`,
the other *must* be an absolute space such as `ShapeScaleSpace`.

::: microsim.schema.space

## Objective Lens

Properties of the objective lens used in the simulation.  Will help
to determine the PSF and other optical properties of the simulation.

::: microsim.schema.lens

## Optical Configs

Optical configurations define the filters and wavelengths used
in the simulation.

::: microsim.schema.optical_config

## Samples

Samples define the distribution and properties of fluorophores in
the simulation.  A `Sample` is a set of `labels`, each of which
is a combination of a `Fluorophore` and a `FluorophoreDistribution`.

The `FluorophoreDistribution` is the primary object that determines
what the sample will look like.

::: microsim.schema.sample

## Modality

Modalities define the imaging modality used in the simulation,
such as "widefield", "confocal", "3D-SIM", etc.  This object
has a lot of control over how other parts of the simulation
are combined to render the final image.

::: microsim.schema.modality

## Settings

Various global settings for the simulation, such as the calculation
backend (`numpy`, `cupy`, `jax`, etc...) and an optional random seed,
which can be used to reproduce the same simulation multiple times.

Note that all of these settings can also be defined using environment
variables prefixed with `MICROSIM_`.  For example, to globally
disallow caching, you can set `MICROSIM_CACHE=0`.  Nested settings
can be defined using `__` as a separator, such as `MICROSIM_CACHE__WRITE=0`,
which would disable writing to the cache (but still allow reading).

::: microsim.schema.settings

## Other Types

Types used in the schema that may be used across categories.

::: microsim.schema.spectrum
