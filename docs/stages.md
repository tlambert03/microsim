# Simulation Stages

The simulation is divided into several stages, each of which is responsible for a specific part of the simulation process. The stages are executed in the following order:

1. [Establishing the 3D space in which the simulation is performed](#space-creation)
2. [Generation of the ground truth target/fluorophore positions](#ground-truth)
3. [Generation of emission photon fluxes](#emission-flux)
4. [Forming the optical image](#optical-image)
5. [Noise and downsampling in the digital image](#digital-image)

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
        },
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

It's worth pointing out that we need to calculate the emission spectral flux for
*every* combination of fluorophore and channel in order to be able to include
bleedthrough and crosstalk effects in the simulation. For example `data[{'F': 0,
'C': 1}]` would represent the emission flux of fluorophore 0 when excited by
channel 1. This is why both F and C remain at this stage.  

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
        ],
    )
    ```

!!! note "TODO"

    Things like fluorophore lifetime, ground state depletion, and photobleaching
    rate also need to be considered in this stage, but are not yet implemented. In
    order to properly simulate these effects, the simulation will need to also take
    into account the temporal aspects of the illumination pattern even *within* a
    single image.  That's a lot of complexity.  So for now, everything will be
    simulated as a steady-state emission flux.

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

You can also load optical configurations from [FPbase
microscope](https://www.fpbase.org/microscopes) using the syntax
`microscope_id::config_name`. For example, to load the "Widefield Green" config
from the [Example Simple Widefield microscope on
FPbase](https://www.fpbase.org/microscope/wKqWbgApvguSNDSRZNSfpN/?c=Widefield%20Green),
you would grab the microscope id from the URL (in this case `wKqWbgAp`) and add
the config name (`Widefield Green`), separated by two colons (`::`):

!!! example

    ```python
    sim = Simulation(
        # ...,
        channels=["wKqWbgAp::Widefield Green"],
    )
    ```

!!! tip

    Here are some useful FPbase microscopes with brand-specific filter set catalogs:

    - [Chroma Filter Sets](https://www.fpbase.org/microscope/PMtA2nB6Ld2Y2XvrF5zUuP/)
    - [Semrock Filter Sets](https://www.fpbase.org/microscope/HGtCWRnyn8joPY5WF2t3zW/)
    - [Nikon Filter Sets](https://www.fpbase.org/microscope/up3K5Tp3jwLWXtoTt8T9vB/)
    - [Zeiss Filter Sets](https://www.fpbase.org/microscope/VgeWjEPrGiSL6saRi9myA8/)

## Optical Image

*i.e. the "filtered" Emission Flux*

- **value units**: photons / second
- **dimensions lost**:
    - **`W`**: wavelength
    - **`F`**: Fluorophore

In this stage, emission photon fluxes are convolved with the optical point
spread function (PSF) of the microscope and wavelengths are filtered based on
the emission path configuration to form the (noise free) optical image.

!!! info "PSF"

    The [point spread function](https://en.wikipedia.org/wiki/Point_spread_function)
    describes the response of the imaging system to a point source. In the context of
    fluorescence microscopy, the PSF describes how light emitted from a point
    emitter (i.e. a single fluorophore) is "spread out" or blurred in the image due
    to diffraction.

    In microsim, the PSF largely be determined by the [`ObjectiveLens`][microsim.schema.ObjectiveLens] and the
    [`Modality`](api.md#modality) of the simulation

The output of this stage is a 4D array with dimensions `(C, Z, Y, X)` where the
wavelength and fluorophore dimensions have been collapsed into the channel. Note
that each channel dimension will contain the emission of *all* fluorophores that
emit in the wavelength range of that channel. This is important for simulating
"crosstalk" or "bleedthrough".  Even for the case of a spectral detection system,
the `W` dimension will be removed, but there would be a channel for each wavelength
bin in the detector.

!!! example

    This example sets up a simulation with a confocal microscope with a 1.4 NA
    objective lens and a 1.2 AU pinhole. The simulation is set up to use the "491"
    optical configuration from the [Example Yokogawa
    Setup](https://www.fpbase.org/microscope/4yL4ggAo/?c=491) (spinning disc confocal)
    microscope on FPbase.

    ```python
    sim = Simulation(
        # ...,
        channels=["4yL4ggAo::491"],
        objective_lens={"numerical_aperture": 1.4, "immersion_medium_ri": 1.515, 'specimen_ri': 1.33},
        modality={"type": "confocal", 'pinhole_au': 1.2},
    )
    ```

## Digital Image

- **value units**: gray levels

The final stage of the simulation is the conversion of the filtered optical
image into a digital image. This stage includes the addition of noise
([photon or shot noise](https://en.wikipedia.org/wiki/Shot_noise) due to the
discrete nature of photons, [read
noise](https://hamamatsu.magnet.fsu.edu/articles/ccdsnr.html) due to imprecision
in the detector, and other sources) as well as downsampling to the final image
resolution.

The output of this stage is a 4D array with dimensions `(C, Z, Y, X)` where the
values are in units of gray levels (integers).  The values in the digital image
will depend on the settings of the `detector` and the `output_space`.

!!! example

    This example sets up a simulation with a 16-bit detector with a read noise of
    2 electrons rms, and a quantum efficiency of 0.82.  Note that, for now,
    the "pixel size" is implicitly determined by the `output_space` parameter.
    But that will change in the future.

    ```python
    sim = Simulation(
        # ...,
        output_space={"downscale": 4},
        detector={"bit_depth": 16, "read_noise": 2, "qe": 0.82},
    )
    ```

!!! note "TODO"

    We don't yet model pixel-specific characterists of sCMOS cameras.  Ideally,
    we will generate a random "instance" of an sCMOS camera with noise, gain,
    and offset distribution pulled from a typical CMOS.
