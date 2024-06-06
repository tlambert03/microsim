# Simulation Stages

The simulation is divided into several stages, each of which is responsible for a specific part of the simulation process. The stages are executed in the following order:

1. [Establishing the 3D space in which the simulation is performed](#space-creation)
2. [Generation of the ground truth target/fluorophore positions](#ground-truth)
3. [Generation of emission photon fluxes](#emission-flux)
4. [Forming the optical image](#optical-image)

## Space creation

- **value units**: N/A
- **dimensions introduced**:
    - **`Z`**: planes along the optical axis (units length).
    - **`Y`**: rows (units length).
    - **`X`**: columns (units length).

The simulation starts with the declaration of a discrete 3D volume in which the
simulation will take place. The volume is defined by its shape and scale (voxel
size in physical units).  Typically, this volume will be many times larger than
the final image size (in terms of shape) so as to be able to represent very fine
ground truth structures that will be blurred by the imaging system. At a later
stage, this volume will be downsampled to the final image shape and scale
(representing the detector pixel sizes).

!!! example

    The following example creates a 3D volume of shape (512, 1024, 1024) with voxel
    sizes of (0.04, 0.02, 0.02) microns (i.e. a volume of 20.48 x 20.48 x 20.48 µm).
    The `output_space` parameter specifies that the final image will be downsampled
    by a factor of 4, into a volume of shape (128, 256, 256) with voxel sizes of
    (0.16, 0.08, 0.08) microns.  Note that the *extents* of the ground truth and
    output volumes are the same.

    ```python
    from microsim import Simulation

    sim = Simulation(
        truth_space={'shape': (512, 1024, 1024), 'scale': (0.04, 0.02, 0.02)},
        output_space={"downscale": 4},
    )
    ```

## Ground Truth

- **value units**: fluorophore counts
- **dimensions introduced**:
    - **`F`**: Fluorophore (categorical unit [`FluorophoreDistribution`][microsim.schema.FluorophoreDistribution])

The ground truth stage is responsible for generating the positions of the
fluorophores in the 3D space. Each
[`FluorophoreDistribution`][microsim.schema.FluorophoreDistribution] object in
the simulation specifies the position, number, species (e.g. mEGFP) of
fluorophores in the volume. The ground truth will have dimensions `(F, Z, Y, X)`
where `len(F)` is determined by the number of `labels` in the `sample` field of
the `Simulation`; the values in the ground truth array represent the number of
fluorophores present at each voxel in the volume.

!!! example

    Here is an example using a ground-truth generator
    [`MatsLines`][microsim.schema.sample.MatsLines] that draws random lines in the
    volume.  The fluorophore is specified to be `EGFP`.
    
    *When specifying the fluorophore as a string, microsim will load properties from
    [FPbase](https://www.fpbase.org/). Otherwise, you may directly create a
    [`Fluorophore`][microsim.schema.Fluorophore] object.*

    ```python
    sim = Simulation(
        # ...,
        sample={
            'labels': [
                {
                    'distribution': {'type': 'matslines', 'density': 0.5, 'length': 30, 'azimuth': 5},
                    'fluorophore': 'EGFP'
                }
            ],
        }
    )
    ```

!!! tip "Have a cool ground truth generator?"

    This is obviously a very important stage, as the quality of the ground truth
    will directly affect the realism of the simulation. We provide several built-in
    ground truth generators, but *we would love to expand this with additional
    user-contributed methods*. Please [open a
    PR](https://github.com/tlambert03/microsim/pulls)!
    We will help :slightly_smiling_face:

## Emission Flux

- **value units**: photons / second
- **dimensions introduced**:
    - **`W`**: wavelength (units length)
    - **`C`**: Channel (categorical unit [`OpticalConfig`][microsim.schema.OpticalConfig])

In this stage, fluorophore counts are converted into emission photon fluxes in
units of photons per second (per voxel). This is still considered to be in the
pre-optical domain, before the effects of the detection optics are applied
(but *after* any effects of the excitation optics are applied).

This conversion will depend on:

- the excitation spectrum of the fluorophore
- the spectrum and irradiance (W/cm^2) of the excitation light source and filters
- the pattern of illumination (particularly relevant for structured illumination)
- the molecular brightness of the fluorophore (extinction coefficient and quantum yield).

The output of this stage is a 6D array with dimensions `(W, C, F, Z, Y, X)`
with two new dimensions:

- The wavelength (`W`) dimension has coordinates representing the wavelength of
  the emitted photons, and `len(W)` is determined by the number of wavelength bins
  in the simulation.
- The channel (`C`) dimension has coordinates representing the different optical
  configurations (e.g. filter sets) in the simulation, and `len(C)` is determined
  by the number of `channels` in the simulation.

!!! examples

    Each channel in the simulation is an arrangement of optical filters
    and light sources.  (If the light source is omitted, it is assumed to be
    a flat white light source, and the excitation filter spectra are used directly).

    ```python
    sim = Simulation(
        # ...,
        channels=[
            {
                "name": "Green",
                "filters": [
                    {"type": "bandpass", "bandcenter": 470, "bandwidth": 40, "placement": "EX"},
                    {"type": "longpass", "cuton": 495, "placement": "BS"},
                    {"type": "bandpass", "bandcenter": 525, "bandwidth": 50, "placement": "EM"}
                ],
            }
        ]
    )
    ```

!!! note "TODO"

    Fluorophore lifetime, ground state depletion, and photobleaching rate
    also need to be considered in this stage, but are not yet implemented.

### Builtin library of common filter sets

`microsim` provides a library of common optical configs.  For example,
the above filter set arrangement shown above is a very common FITC filter set,
which can be loaded from the library as follows:

!!! example

    ```python
    from microsim.schema.optical_config import lib

    sim = Simulation(
        # ...,
        channels=[lib.FITC],
    )
    ```

### Optical Configurations from FPbase

You can also load optical configurations from any microscope created in
[FPbase](https://www.fpbase.org/) using the syntax `microscope_id::config_name`.
For example, to load the "Widefield Green" config from the [Example Simple
Widefield microscope on
FPbase](https://www.fpbase.org/microscope/wKqWbgApvguSNDSRZNSfpN/), you would
grab the microscope id from the URL (in this case `wKqWbgAp`) and add the
config name (`Widefield Green`), separated by two colons (`::`):

!!! example

    ```python
    sim = Simulation(
        # ...,
        channels=["wKqWbgAp::Widefield Green"],
    )
    ```

Here are some useful microscopes with brand-specific filter set catalogs:

- [Chroma Filter Sets](https://www.fpbase.org/microscope/PMtA2nB6Ld2Y2XvrF5zUuP/)
- [Semrock Filter Sets](https://www.fpbase.org/microscope/HGtCWRnyn8joPY5WF2t3zW/)
- [Nikon Filter Sets](https://www.fpbase.org/microscope/up3K5Tp3jwLWXtoTt8T9vB/)
- [Zeiss Filter Sets](https://www.fpbase.org/microscope/VgeWjEPrGiSL6saRi9myA8/)

## Optical Image

- **value units**: photons / second
- **dimensions lost**:
    - **`W`**: wavelength
    - **`F`**: Fluorophore

In this stage, the emission photon fluxes are convolved with the optical point
spread function (PSF) of the microscope to form the (noise free) optical image.
