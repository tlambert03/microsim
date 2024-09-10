# Simulation Stages

The simulation is divided into several stages, each of which is responsible for a specific part of the simulation process. The stages are executed in the following order:

1. [Establishing the 3D space in which the simulation is performed](#space-creation)
2. [Generation of the ground truth target/fluorophore positions](#ground-truth)
3. [Calculation of emission photon fluxes](#fluorophore-emission-collection)
4. [Forming the optical image](#optical-image)
5. [Noise and downsampling in the digital image](#digital-image)

## Space creation

- **value units**: N/A
- **dimensions**:
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

- **method**: [`Simulation.ground_truth`][microsim.Simulation.ground_truth]
- **value units**: fluorophore counts
- **dimensions**:
    - **`F`**: Fluorophore (categorical unit [`Fluorophore`][microsim.schema.Fluorophore])
    - **`Z`**: planes along the optical axis (units length).
    - **`Y`**: rows (units length).
    - **`X`**: columns (units length)

The ground truth stage is responsible for generating the positions and number of
the fluorophores in the 3D space. Each
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

## Fluorophore Emission & Collection

- **method**: [`Simulation.filtered_emission_rates`][microsim.Simulation.filtered_emission_rates]
- **value units**: photons / second
- **dimensions**:
    - **`C`**: Channel (categorical unit [`OpticalConfig`][microsim.schema.OpticalConfig])
    - **`F`**: Fluorophore (categorical unit [`Fluorophore`][microsim.schema.Fluorophore])
    - **`W`**: wavelength (units length)

In this stage, we calculate the rate of photon emission & collection for each
combination of fluorophore and optical configuration in the simulation, as a
function of wavelength. This is still considered to be in the "pre-optical"
domain, before the effects of the detection optics are applied, but *after* any
effects of the emission optics and QE are applied.  (i.e. it's slightly out of
order, but it allows us to more accurately calculate wavelength effects).

This array is *not* spatially aware, but will be combined with the ground truth
array of fluorophore concentration to determine the spatial distribution of
emitted photons.

This stage includes a number of calculations, and will generally depend on:

- the excitation spectrum of the fluorophore
- the spectrum and irradiance (W/cm^2) of the excitation light source and filters
- the molecular brightness of the fluorophore (extinction coefficient and quantum yield).
- the spectral transmission of the emission filters and detector QE.

### Absorption cross section

The absorption cross section $\sigma$ (in $cm^2$) of the fluorophore at
each wavelength is given by:

$$
\sigma(\lambda) = \log(10) \frac{\epsilon(\lambda) \cdot 10^3}{N_A}
$$

where:

- $\epsilon(\lambda)$ is the Molar extinction coefficient at each wavelength ($\text{M}^{-1} \text{cm}^{-1}$)
- $N_A$ is Avogadro's number ($6.022 \times 10^{23} / \text{mol}$)

### Irradiance Flux Density

The flux of excitation photons $\Phi_{\text{ex}}$ (in $photons/cm^2/sec$)
at each wavelength is given by:

$$
\Phi_{\text{ex}}(\lambda) = \frac{P(\lambda) \cdot \lambda}{hc}
$$

where:

- $P(\lambda)$ The spectrum of the light source and its irradiance ($W/cm^2$)
- $h$ is Planck's constant ($6.626 \times 10^{-34} J \cdot s$)
- $c$ is the speed of light ($3 \times 10^8$ $m/s$)
- $\lambda$: the wavelength of the excitation photons ($m$).

*Note that in the simulation, $P(\lambda)$ will also include the effects of the
excitation filters, which will reduce the irradiance at specific wavelengths.*

### Absorption Rate

The effective rate of photon absorption $\Phi_{\text{abs}}$ (in $photons/sec$)
at each wavelength is the product of the excitation flux and the absorption cross
section:

$$
\Phi_{\text{abs}}(\lambda) = \Phi_{\text{ex}}(\lambda) \times \sigma(\lambda)
$$

Summed over all wavelengths, this gives the total absorption rate $\Phi_{\text{abs,total}}$:

$$
\Phi_{\text{abs,total}} = \sum_{\lambda} \Phi_{\text{abs}}(\lambda)
$$

### Emission Rate

To convert the absorption rate into an emission rate, we need to consider the
quantum yield $\eta$ of the fluorophore, and the emission spectrum of the
fluorophore. First, we normalize the emission spectrum $I_{\text{em}}(\lambda)$
so that the integral over all wavelengths is 1:

$$
   \tilde{I}_{\text{em}}(\lambda) = \frac{I_{\text{em}}(\lambda)}{\sum_{\lambda} I_{\text{em}}(\lambda)}
$$

The emission rate $\Phi_{\text{em}}$ (in $photons/sec$) at each
wavelength is calculated by multiplying the normalized emission spectrum by the
absorption rate and the quantum yield $\eta$:

$$
   \Phi_{\text{em}}(\lambda) = \eta \times \Phi_{\text{abs, total}} \times \tilde{I}_{\text{em}}(\lambda)
$$

where:

- $\eta$ is the quantum yield of the fluorophore.

### Filtered Emission Flux

The emission rate $\Phi_{\text{em}}$ is then multiplied by the combined
transmission of the emission filters and the quantum efficiency of the detector
to give the final emission rate $\Phi_{\text{em, filtered}}$:

$$
\Phi_{\text{em, filtered}}(\lambda) = \Phi_{\text{em}}(\lambda) \times T_{\text{em}}(\lambda) \times \text{QE}(\lambda)
$$

where:

- $T_{\text{em}}(\lambda)$ is the transmission of the emission filters at each wavelength.
- $\text{QE}(\lambda)$ is the quantum efficiency of the detector at each wavelength.

The output of this stage (given by
[`Simulation.filtered_emission_rates`][microsim.Simulation.filtered_emission_rates])
is a 3D array with dimensions `(C, F, W)`, but lacking spatial information:

- The channel (`C`) dimension has coordinates representing the different optical
  configurations (e.g. filter sets) in the simulation, and `len(C)` is determined
  by the number of `channels` in the simulation.
- The fluorophore (`F`) dimension has coordinates representing the different
  fluorophores in the simulation, and `len(F)` is determined by the number of
  unique fluorophores in the `sample.labels` field of the `Simulation`.
- The wavelength (`W`) dimension has coordinates representing the wavelength of
  the emitted photons, with nanometer resolution.

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

!!! info "Builtin library of common filter sets"

    `microsim` provides a library of common optical configs.  For example,
    the above filter set arrangement shown above is a very common FITC filter set,
    which can be loaded from the library as follows:

    ```python
    from microsim.schema.optical_config import lib

    sim = Simulation(
        # ...,
        channels=[lib.FITC],
    )
    ```

!!! info "Optical Configurations from FPbase"

    You can also load optical configurations from [FPbase
    microscope](https://www.fpbase.org/microscopes) using the syntax
    `microscope_id::config_name`. For example, to load the "Widefield Green" config
    from the [Example Simple Widefield microscope on
    FPbase](https://www.fpbase.org/microscope/wKqWbgApvguSNDSRZNSfpN/?c=Widefield%20Green),
    you would grab the microscope id from the URL (in this case `wKqWbgAp`) and add
    the config name (`Widefield Green`), separated by two colons (`::`):

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

- **method**: [`Simulation.optical_image`][microsim.Simulation.optical_image]
- **value units**: photons / second
- **dimensions**:
    - **`C`**: Channel (categorical unit [`OpticalConfig`][microsim.schema.OpticalConfig])
    - **`Z`**: planes along the optical axis (units length).
    - **`Y`**: rows (units length).
    - **`X`**: columns (units length)

In this stage, the fluorophore distributions are scaled by the respective
emission photon fluxes, and then convolved with the optical point spread
function (PSF) of the microscope to form the (noise free) optical image.

Within each channel, the contributions of individual fluorophores are summed
together to form the final image (allowing for crosstalk and bleedthrough).  If
you want access to the individual fluorophore contributions, you can use
[`Simulation.optical_image_per_fluor`][microsim.Simulation.optical_image_per_fluor],
which retains the full `(C, F, Z, Y, X)` dimensions.

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

- **method**: [`Simulation.digital_image`][microsim.Simulation.digital_image]
- **value units**: gray levels
- **dimensions**:
    - **`C`**: Channel (categorical unit [`OpticalConfig`][microsim.schema.OpticalConfig])
    - **`Z`**: planes along the optical axis (units length).
    - **`Y`**: rows (units length).
    - **`X`**: columns (units length)

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

    We don't yet model pixel-specific characteristics of sCMOS cameras.  Ideally,
    we will generate a random "instance" of an sCMOS camera with noise, gain,
    and offset distribution pulled from a typical CMOS.
