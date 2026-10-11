---
icon: lucide/aperture
---

# Modalities and the PSF

The modality decides how the optical image is formed from the ground truth.
For the modalities available now, that means choosing a point spread function
(PSF) and convolving each fluorophore's emission with it.

```py
sim = ms.Simulation(..., modality=ms.Confocal(pinhole_au=1))
```

| Modality                                   | PSF                                              | Image formed by        |
| ------------------------------------------ | ------------------------------------------------ | ---------------------- |
| [`Widefield`][microsim.schema.Widefield] (default) | emission PSF                              | camera                 |
| [`Confocal`][microsim.schema.Confocal]     | excitation PSF × (emission PSF ⊗ pinhole)        | point scanning         |
| [`Identity`][microsim.schema.Identity]     | none                                             | (ground truth is already an image) |

![The same sample in widefield and confocal](../images/generated/modalities.png)
/// caption
The same random lines in widefield and in confocal with a 1 AU pinhole (XY
plane through the middle of the volume, and an XZ section). The confocal
removes most of the out-of-focus light.
///

## Widefield

The whole field is illuminated uniformly and imaged onto a camera. Each
fluorophore's emission is convolved with the emission PSF. Every plane of the
PSF sums to about 1, so a defocused point source spreads its light over more
pixels without losing it.

## Confocal

A focused laser spot is scanned across the sample, and the emission passes a
pinhole before reaching a point detector. The confocal PSF is the product of

- the **excitation PSF**: the relative irradiance of the focused spot, 1 at
  the center, and
- the **effective detection PSF**: the emission PSF convolved with the pinhole,
  which is the fraction of collected light that passes the pinhole for an
  emitter at each position.

`pinhole_au` is the pinhole diameter in Airy units, where 1 AU is
1.22 λ<sub>em</sub> / NA, measured in the sample. Smaller pinholes reject more
out-of-focus light and slightly improve resolution, at the cost of signal.

For a point-scanning modality:

- `power` on the channel is the **peak irradiance** at the center of the spot
  (often kW/cm² or more), not an average over the field.
- `exposure_ms` is the **pixel dwell time** (often 0.0005–0.01 ms), and is not
  divided among pixels.
- [Saturation](photometry.md#excitation-saturation) is applied at each position
  of the excitation spot rather than to the emission rate.
- Downsampling to the output space averages rather than sums
  ([Spaces](spaces.md#how-the-downsampling-works)).
- A point detector ([PMT or HyD](detectors.md#point-detectors)) is the natural
  choice, though any detector works.

![Widefield and confocal PSFs](../images/generated/psf-comparison.png)
/// caption
Widefield and confocal PSFs at 1.4 NA, 525 nm emission and 488 nm excitation,
in XY (top) and XZ (bottom). Square-root display. With the pinhole closed
to 0.25 AU, the confocal PSF approaches the product of the excitation and
emission PSFs.
///

!!! info "Excitation wavelength"

    In a simulation, the confocal excitation PSF is currently computed at the
    emission wavelength, not at the wavelength of the light source. This makes
    the simulated excitation spot slightly larger than it should be (by the
    ratio of the wavelengths, about 7% for GFP).

## Identity

[`Identity`][microsim.schema.Identity] applies no PSF. Use it when the ground
truth is already a microscope image (for example, a real widefield image) and
you are interested in the spectral and detector parts of the simulation:
fluorophores, filters, bleedthrough, and noise. It must be given as an object,
`modality=ms.Identity()`.

## Objective lens

[`ObjectiveLens`][microsim.schema.ObjectiveLens] sets the numerical aperture
and the refractive indices used by the PSF model. The default is a 1.4 NA oil
immersion objective.

```py
ms.Simulation(
    ...,
    objective_lens=ms.ObjectiveLens(
        numerical_aperture=1.2,
        immersion_medium_ri=1.333,
        immersion_medium_ri_spec=1.333,
        specimen_ri=1.333,
    ),
)
```

| Parameter                     | Default | Meaning                                          |
| ----------------------------- | ------- | ------------------------------------------------ |
| `numerical_aperture`          | 1.4     | NA; cannot exceed the immersion refractive index |
| `immersion_medium_ri`         | 1.515   | refractive index of the immersion medium         |
| `specimen_ri`                 | 1.47    | refractive index of the sample                   |
| `coverslip_ri`                | 1.515   | refractive index of the coverslip                |
| `coverslip_thickness_um`      | 170     | coverslip thickness                              |
| `*_spec` versions             | same    | the values the objective was designed for        |
| `working_distance_um`         | 150     | design working distance                          |

The `*_spec` parameters describe the conditions the objective is corrected
for. When the actual values differ (a coverslip of the wrong thickness, or
the wrong immersion oil), the PSF shows the resulting spherical aberration.

The NA also sets the [collection efficiency](photometry.md) (the fraction of
emitted light that enters the objective), so a lower NA gives a dimmer
image as well as a larger PSF.

??? note "The PSF model"

    PSFs are computed with a vectorial model of a high-NA objective that
    includes the Gibson–Lanni optical path difference for mismatched
    coverslip, immersion, and specimen refractive indices
    ([`microsim.psf`](../api/psf.md)).

    The same PSF is used for the whole volume (it is spatially invariant), and
    is computed for a point source at the coverslip surface. Aberrations that
    grow with imaging depth into a mismatched specimen are therefore not
    included.

## Emission spectrum and the PSF

The PSF depends on wavelength, and a fluorophore emits over a range of
wavelengths. By default, microsim computes one PSF per fluorophore and
channel, at the centroid wavelength of the detected emission. For a more
accurate result, it can split the detected emission into several wavelength
bins and sum one PSF per bin, weighted by the photons in that bin:

```py
ms.Settings(spectral_bins_per_emission_channel=3)
```

This takes proportionally longer. See [Settings](settings.md#accuracy-vs-speed).

## PSF caching

PSFs are cached in memory and on disk, keyed by every parameter that affects
them, so repeating a simulation with a different sample or detector does not
recompute them. See [Settings](settings.md#caching).
