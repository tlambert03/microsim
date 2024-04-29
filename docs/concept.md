# Conceptual Overview

The goal of microsim is to generate highly realistic simulated light microscopy
images. The top level `microsim.Simulation` object contains all of the
parameters that define the simulation, such as the shape and scale of the space
in which the simulation is run, the sample to be imaged, the modality and
optical configuration of the microscope, the detector, and the shape/scale of
the final output image. Parameters are designed to be swappable and extensible,
so it should be easy to "image" the same sample on various microscope setups.

Properties of fluorophores and optical configurations for microscopes can be
loaded from external sources such as [FPbase](https://www.fpbase.org/).

The simulation can be run on a CPU or GPU, and can be accelerated with Jax,
Cupy, or PyTorch.

Possible applications include:

- Teaching microscopy concepts
- Generating realistic training data for machine learning models
- Generating test data for image processing pipelines
- Generating challenge/benchmark datasets with known ground truth.

## Simulation Parameter Objects

Microsim is designed as a set of pydantic objects.  In addition to
runtime type checking and coercion, this allows for easy serialization and
deserialization of the simulation parameters.

### Simulation

All simulations start with a [`microsim.Simulation`][] object, and the
simulation is run by calling the [`run()`][microsim.Simulation.run] method on this
object.

### Spaces

Spaces define the shape and scale of the space in which the simulation is run.
They can either be "concrete" spaces that provide the actual shape and scale,
or relative spaces that provide a shape and scale downsampled or upsampled
relative to another space.

### Samples

Samples are declared as a list of
[`FluorophoreDistribution`][microsim.schema.FluorophoreDistribution] objects,
each of which specifies the distribution of fluorophores in the sample, and the
type of fluorophore (e.g. EGFP, mCherry, etc.). Changing the fluorophore may
change spectral properties, or signal to noise ratio, while changing the
distribution changes the spatial properties of the sample.

### Modalities

The modality controls much of the image formation process, its `render()` method
does much of the heavy lifting in the simulation.  The modality is responsible
for generating the raw image data from ground truth fluorophore positions.
Example modalities include confocal, widefield, structured illumination, etc...

### Objective Lenses

Objective lenses control the properties of the objective lens, such as numerical
aperture.  Currently, this object is a bit overloaded as it also includes things
like coverslip thickness.

### Channels

Channels control the arrangement of filters in the microscope: excitation, emission,
and dichroic filters.  Channels can be used by the modality to determine signal,
background, bleedthrough, etc...

### Detectors

Detectors control the properties of the detector, such as quantum efficiency,
dark current, read noise, etc... The detector properties are generally applied
after downsampling the image to the output (pixelated) space.

### Settings

The settings object controls various global parameters of the simulation, such
as the numpy-like backend to use (numpy, jax, cupy, pytorch), a random seed which
can be used to make the simulation reproducible, and other general parameters.

## Additional Functionality

### Point Spread Functions

There is no top-level point spread function object in the simulation: most
modalities will be responsible for generating their own point spread function
based on the objective lens and other parameter that are passed into their
`render()` method.  The `microsim.psf` module contains some common point spread
functions, with `make_psf()` being the most common entrypoint.

### FPbase

The `microsim.fpbase` module contains functions for loading fluorophore and
optical configuration data from [FPbase](https://www.fpbase.org/).

### Open organelle

The `cosem` folder contains functions for loading data from
<https://openorganelle.janelia.org>, which is an incredibly rich source of
ground truth data.  Open organelle has a database of FIB-SEM data collected
at 4nm resolution, and segmented into organelles.  (This is a work in progress)
