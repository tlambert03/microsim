---
icon: lucide/filter
---

# Channels

A channel is one optical configuration of the microscope: the light source
and the filters it uses to excite the sample and to select the emitted light.
`Simulation.channels` is a list of [`OpticalConfig`][microsim.schema.OpticalConfig]
objects, and the output has one image per channel along the `c` dimension.

The default is a single FITC (GFP) channel.

```py
from microsim.schema.optical_config import lib

sim = ms.Simulation(..., channels=[lib.FITC, lib.TRITC])
```

<div class="ms-chart" data-chart="spectra"></div>
/// caption
A fluorophore's absorption and emission spectra (normalized to their peaks), the
excitation light that reaches the sample, and the transmission of the emission
path. The emission path combines every filter in it: the dichroic mirror
reflects the excitation light toward the sample and transmits the emission to
the detector. Shaded: the absorption spectrum weighted by the excitation light,
and the emission that passes the emission path. Hover for values.
///

## Built-in filter sets

[`microsim.schema.optical_config.lib`][microsim.schema.optical_config.lib]
has idealized versions of common filter sets, built from smooth bandpass and
longpass edges. They are not models of real filters, and are meant for quick
simulations; for real filters, load them from FPbase (below).

| Name    | Excitation  | Dichroic | Emission   |
| ------- | ----------- | -------- | ---------- |
| `DAPI`  | 350/50      | 400 LP   | 460/50     |
| `ECFP`  | 436/20      | 455 LP   | 480/40     |
| `FITC`  | 470/40      | 495 LP   | 525/50     |
| `EYFP`  | 500/20      | 515 LP   | 535/30     |
| `TRITC` | 545/25      | 565 LP   | 605/70     |
| `DSRED` | 545/30      | 570 LP   | 620/60     |
| `CY5`   | 620/60      | 660 LP   | 700/75     |
| `CY7`   | 710/75      | 760 LP   | 810/90     |

Bandpass filters are written as center/width in nm.

## Filter sets from FPbase

FPbase lets you
[enter the optical configurations of your own microscope](https://www.fpbase.org/microscope/create/),
with measured spectra of its light sources, filters, and detectors, and has
[many microscopes](https://www.fpbase.org/microscopes/) entered by others.
These configurations can be loaded in microsim with a string of the form
`"<microscope id>::<config name>"`:

```py
sim = ms.Simulation(..., channels=["wKqWbg::Widefield Green"])
# equivalent:
ms.OpticalConfig.from_fpbase("wKqWbg", "Widefield Green")
```

The microscope ID is in the URL of its FPbase page, e.g.
[`fpbase.org/microscope/wKqWbgApvguSNDSRZNSfpN/`](https://www.fpbase.org/microscope/wKqWbgApvguSNDSRZNSfpN/) (the first few characters are
enough). Configuration names are not case-sensitive. Some FPbase microscopes
with filter catalogs:

- [Chroma filter sets](https://www.fpbase.org/microscope/PMtA2nB6Ld2Y2XvrF5zUuP/)
- [Semrock filter sets](https://www.fpbase.org/microscope/HGtCWRnyn8joPY5WF2t3zW/)
- [Nikon filter sets](https://www.fpbase.org/microscope/up3K5Tp3jwLWXtoTt8T9vB/)
- [Zeiss filter sets](https://www.fpbase.org/microscope/VgeWjEPrGiSL6saRi9myA8/)

## Building your own

```py
ms.OpticalConfig(
    name="GFP, 488 laser",
    lights=[ms.LightSource.laser(488)],
    filters=[
        ms.Longpass(cuton=495, placement="BS"),
        ms.Bandpass(bandcenter=525, bandwidth=50, placement="EM"),
    ],
    power=2000,  # W/cm² at the sample
)
```

### Filters

| Filter                                              | Parameters                                                |
| --------------------------------------------------- | --------------------------------------------------------- |
| [`Bandpass`][microsim.schema.Bandpass]              | `bandcenter`, `bandwidth` (nm), peak `transmission`        |
| [`Longpass`][microsim.schema.Longpass]              | `cuton` (nm), edge `slope`, `transmission`                 |
| [`Shortpass`][microsim.schema.Shortpass]            | `cutoff` (nm), edge `slope`, `transmission`                |
| [`SpectrumFilter`][microsim.schema.SpectrumFilter]  | a measured `transmission` [`Spectrum`][microsim.schema.Spectrum] |

`placement` says where a filter sits in the light path:

| `placement` | Excitation path         | Emission path           | Typical use                    |
| ----------- | ----------------------- | ----------------------- | ------------------------------ |
| `"EX"`      | transmission            | (not used)              | excitation filter              |
| `"EM"`      | (not used)              | transmission            | emission filter                |
| `"BS"`      | 1 − transmission        | transmission            | dichroic that reflects excitation |
| `"BSi"`     | transmission            | 1 − transmission        | dichroic that reflects emission |
| `"ALL"`     | transmission            | transmission            | e.g. a neutral density filter  |

The defaults are `"EX"` for `Shortpass`, `"EM"` for `Longpass`, and `"ALL"`
for `Bandpass` and `SpectrumFilter`, so set `placement` explicitly for
bandpass filters. Filters in the same path are multiplied together.

### Light sources

`lights` is a list of [`LightSource`][microsim.schema.LightSource] objects,
each with an emission `spectrum`. [`LightSource.laser(488)`][microsim.schema.LightSource.laser]
makes a narrow line at 488 nm. With no light source, the channel is
illuminated by flat white light, so the excitation filters alone set the
spectrum.

### Illumination power

`OpticalConfig.power` is the irradiance at the sample in W/cm², summed over
all wavelengths after the excitation filters. If it is not set, the irradiance
is 1 W/cm².

!!! warning "`LightSource.power` sets relative power only"

    When a channel has several light sources, their `power` values set how the
    total is divided between them. The total is always
    `OpticalConfig.power` (or 1 W/cm²). Set `power` on the
    `OpticalConfig` to change brightness.

Rough orders of magnitude for the irradiance at the sample:

| Microscope                          | Irradiance              |
| ----------------------------------- | ----------------------- |
| widefield, arc lamp or LED          | 0.1–10 W/cm²            |
| widefield, laser                    | 10–1000 W/cm²           |
| point-scanning confocal, at focus   | 1–1000 kW/cm² and above |

For point-scanning modalities, `power` is the peak irradiance at the center of
the focused spot. See [Modalities](modalities.md#confocal).

### Per-channel exposure

`OpticalConfig.exposure_ms` overrides `Simulation.exposure_ms` for one
channel:

```py
channels = [
    lib.FITC.model_copy(update={"exposure_ms": 50}),
    lib.TRITC.model_copy(update={"exposure_ms": 200}),
]
```

!!! tip "Modifying the library configurations"

    The objects in `lib` are shared. Use `model_copy(update={...})` to make a
    modified copy, rather than assigning to `lib.FITC.power` directly, which
    would change it everywhere in your session.

## Bleedthrough and cross-excitation

microsim computes every fluorophore in every channel, so crosstalk needs no
extra setup. A fluorophore excited by another channel's light
(**cross-excitation**), or whose emission passes another channel's filter
(**bleedthrough**), appears in that channel in proportion to the overlap of the
spectra.

<div class="ms-chart" data-chart="crosstalk"></div>
/// caption
Detected photons per molecule for two fluorophores in two channels. Dashed
cells are crosstalk: the fluorophore gives less than half of that channel's
signal. tdTomato absorbs only weakly in the FITC excitation band, but it is
bright, and a 500 nm longpass emission filter passes nearly all of its
emission: per molecule, it gives almost as much signal in the green channel as
EGFP. With the 525/50 bandpass emission filter of `FITC`, its share drops below
1%. Select `FITC` as channel 1 to compare.
///

To see where the signal in each channel comes from in an image, use
[`optical_image_per_fluor()`][microsim.Simulation.optical_image_per_fluor],
which keeps the fluorophore dimension:

```py
sim = ms.Simulation(
    truth_space=ms.ShapeScaleSpace(shape=(64, 512, 512), scale=0.064),
    sample=[
        ms.FluorophoreDistribution(
            distribution=ms.CosemLabel(dataset="jrc_hela-3", label="er-mem_pred"),
            fluorophore="EGFP",
        ),
        ms.FluorophoreDistribution(
            distribution=ms.CosemLabel(dataset="jrc_hela-3", label="mito-mem_pred"),
            fluorophore="tdTomato",
        ),
    ],
    channels=[lib.FITC, fitc_longpass],  # (1)!
)
per_fluor = sim.optical_image_per_fluor()  # (c, f, z, y, x)
per_fluor.sum(["z", "y", "x"])  # total photons/s from each fluorophore in each channel
```

1.  `fitc_longpass` is `lib.FITC` with its emission filter replaced by
    `ms.Longpass(cuton=500, placement="EM")`.

![EGFP and tdTomato in two green channels](../images/generated/crosstalk.png)
/// caption
A HeLa cell with EGFP on the ER and tdTomato on the mitochondria, recorded in
the green channel with a 525/50 bandpass (top) and a 500 nm longpass (bottom)
emission filter. Each row has one display range, so each panel shows its true
contribution. With the longpass filter, the mitochondria appear in the green
channel.
///

## Plotting spectra

These need the `view` extra. For one channel:

```py
oc = ms.OpticalConfig.from_fpbase("wKqWbg", "Widefield Green")
oc.plot()  # all filter spectra
```

![Output of OpticalConfig.plot](../images/generated/plot-config.png){ .plot }

```py
fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
oc.plot_excitation(
    ax=axes[0]
)  # light source and excitation filters, and the photon flux
oc.plot_emission(ax=axes[1])  # emission filters, and their combined transmission
```

![Output of plot_excitation and plot_emission](../images/generated/plot-paths.png){ .plot }

For a whole simulation, [`Simulation.plot()`][microsim.Simulation.plot] shows,
for each channel (columns): the absorption cross sections, the excitation
path, the absorption rates (with the emission spectra, dashed), the emission
path, and the photons collected from each fluorophore. Here EGFP and mCherry
in the `FITC` and `TRITC` channels:

```py
sim.plot()
```

![Output of Simulation.plot](../images/generated/plot-simulation.png){ .plot }

## Spectral detection

[`lib.spectral_detector`][microsim.schema.optical_config.lib.spectral_detector]
returns a list of channels that split an emission range into equal bins, as
on a spectral confocal:

```py
channels = lib.spectral_detector(bins=8, min_wave=480, max_wave=600, lasers=[488, 561])
```

This is a simplified model: each bin is an ideal rectangular filter, and a
dichroic reflects a 10 nm band around each laser line.
