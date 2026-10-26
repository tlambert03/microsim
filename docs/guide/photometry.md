---
icon: lucide/sigma
---

# Photon budget

This page explains how microsim calculates the number of photons a fluorophore
emits and how many of them reach the detector. These calculations happen in
[`filtered_emission_rates()`][microsim.schema.simulation.Simulation.filtered_emission_rates],
before anything spatial is computed.

<div class="ms-chart" data-chart="photon-budget"></div>
/// caption
Photons absorbed, emitted, and detected per second by one molecule, calculated
as microsim does, for a specimen refractive index of 1.47 (the
`ObjectiveLens` default). With the defaults (EGFP, FITC channel, 1 W/cm²,
1.4 NA, no detector QE), one molecule gives about 6 photons in a 100 ms
exposure. Typical widefield irradiance is 0.1–10 W/cm²; the peak irradiance of
a confocal spot is 10–1000 kW/cm²
([Channels](channels.md#illumination-power)). Fluorophores without a known
lifetime do not saturate.
///

## In short

For each fluorophore in each channel:

1. The light source and excitation filters give a photon flux at the sample.
2. The fluorophore's absorption cross section turns that into an **absorption
   rate**.
3. The quantum yield and emission spectrum turn that into an **emission
   rate**, per wavelength.
4. The emission filters, the detector's QE, and the fraction of light the
   objective collects reduce it to the **detected rate**.

The result is an array with dimensions `(c, f, w)`: photons per second, per
fluorophore, for every channel, fluorophore, and wavelength. Keeping the
wavelength lets the next stage use a PSF for the actual emission color.
Keeping every channel–fluorophore pair is what produces bleedthrough.

These rates are for a fluorophore at the channel's `power`: anywhere in a
widefield field of view, or at the center of the focused spot in a confocal.
By default they include [excitation saturation](#excitation-saturation) at
that irradiance.

You can compute each step on its own:

```py
oc, fluor = lib.FITC, ms.Fluorophore.from_fpbase("EGFP")

oc.illumination_flux_density  # photons/cm²/s, per wavelength
fluor.absorption_cross_section  # cm², per wavelength
oc.absorption_rate(fluor)  # photons/s absorbed
oc.total_emission_rate(fluor)  # photons/s emitted
oc.filtered_emission_rate(fluor)  # photons/s through the emission path
sim.filtered_emission_rates()  # all pairs, with QE and collection efficiency
sim.filtered_emission_rates(saturate=False)  # the same, without saturation
```

The `OpticalConfig` methods describe a channel on its own. `Simulation` also
accounts for channels that are [acquired at the same
time](#channels-acquired-together).

## The details

??? abstract "1. Absorption cross section"

    The excitation spectrum from FPbase (or your
    [`Fluorophore`](samples.md#fluorophores)) is normalized to 1 at its peak and
    multiplied by the extinction coefficient \(\varepsilon\) to give
    \(\varepsilon(\lambda)\) in M⁻¹ cm⁻¹. The absorption cross section of one
    molecule, in cm², is

    \[
    \sigma(\lambda) = \frac{\ln(10) \cdot 10^3 \cdot \varepsilon(\lambda)}{N_A}
    \]

    where \(N_A\) is Avogadro's number. For EGFP
    (\(\varepsilon = 55{,}900\) M⁻¹ cm⁻¹) the peak cross section is about
    \(2.1 \times 10^{-16}\) cm².

??? abstract "2. Illumination"

    The illumination spectrum is the light source spectrum (the sum of the
    light sources, weighted by their relative `power`), multiplied by the
    transmission of every filter in the excitation path at the light source's
    wavelengths, so a laser stays a single line. With no light source, it is
    flat white light through the excitation filters. It is scaled so that its
    sum over wavelength equals
    the channel's `power`, \(P\) (1 W/cm² if not set). This gives the
    irradiance \(E(\lambda)\) in W/cm² per wavelength bin, which is converted to
    a photon flux density, in photons/cm²/s:

    \[
    \Phi(\lambda) = \frac{E(\lambda) \, \lambda}{h c}
    \]

??? abstract "3. Absorption rate"

    The rate at which one molecule absorbs photons is the overlap of the cross
    section and the photon flux, summed over the wavelengths of the
    illumination (the cross section is interpolated there):

    \[
    k = \sum_\lambda \sigma(\lambda) \, \Phi(\lambda)
    \]

??? abstract "4. Emission rate"

    The emission spectrum \(I_\text{em}(\lambda)\) is normalized to unit sum,
    and scaled by the quantum yield \(\phi\) and the absorption rate:

    \[
    R_\text{em}(\lambda) = \phi \, k \, \frac{I_\text{em}(\lambda)}{\sum_{\lambda'} I_\text{em}(\lambda')}
    \]

    (With [saturation](#excitation-saturation), \(k\) is replaced by
    \(k / (1 + k\tau)\).)

??? abstract "5. Emission path and detector"

    The emitted light is multiplied by the combined transmission
    \(T_\text{em}(\lambda)\) of every filter in the emission path, and by the
    detector's quantum efficiency \(\text{QE}(\lambda)\), if a detector is set,
    both interpolated at the wavelengths of the emission spectrum:

    \[
    R_\text{det}(\lambda) = R_\text{em}(\lambda) \, T_\text{em}(\lambda) \, \text{QE}(\lambda)
    \]

    QE is applied here, per wavelength, rather than in the detector model.
    The optical image therefore depends on the detector, so if you change to
    a detector with a different QE, compute the optical image again.
    ([#152](https://github.com/tlambert03/microsim/issues/152) tracks this.)

??? abstract "6. Collection efficiency"

    A fluorophore emits in all directions, and the objective collects only the
    light within its acceptance cone. The fraction collected is the solid angle
    of the cone divided by that of a sphere:

    \[
    \eta = \frac{1 - \cos\theta}{2}, \qquad \sin\theta = \min\left(\frac{\text{NA}}{n_s}, 1\right)
    \]

    where \(n_s\) is the refractive index of the specimen. The angle is measured
    in the specimen, where the light is emitted. For a 1.4 NA objective and
    \(n_s = 1.47\), \(\eta = 0.35\); for a 0.8 NA objective and \(n_s = 1.33\), 0.10.

    This ignores reflection losses at interfaces, the dipole emission pattern,
    supercritical-angle fluorescence near the coverslip, and the transmission
    of the objective and other optics.

## Channels acquired together

When channels are acquired at the same time (in one
[track](channels.md#sequential-and-simultaneous-acquisition)), all of their
light sources are on, so a fluorophore absorbs from all of them. Its
absorption rate is then the sum over the light sources of the track, each at
its own channel's `power`:

\[
k = \sum_\text{sources} \sum_\lambda \sigma(\lambda) \, \Phi_\text{source}(\lambda)
\]

The same emission is then detected by each channel of the track, through its
own emission path. A light source shared by several channels of the track is
counted once. With the default, `tracks="sequential"`, each channel's \(k\)
comes from its own light sources only.

## Excitation saturation

A fluorophore that has absorbed a photon cannot absorb another until it
returns to the ground state, on average after its fluorescence lifetime
\(\tau\). At high irradiance, emission therefore grows more slowly than the
excitation. For a two-level system at steady state, the emission rate is
proportional to

\[
\frac{k}{1 + k\tau}
\]

The product \(s = k\tau\) is the saturation parameter
([`OpticalConfig.saturation_parameter`][microsim.schema.optical_config.OpticalConfig.saturation_parameter],
for a channel on its own; in a track, \(k\) includes every light source of
the track).
When \(s \ll 1\), emission is proportional to irradiance. At \(s = 1\) it is
half of what a linear response would give.

<div class="ms-chart" data-chart="saturation"></div>
/// caption
Emission rate of one molecule against irradiance, uniform over the molecule as
in widefield. The rate approaches \(\phi / \tau\), one photon per lifetime
times the quantum yield.
///

How saturation is applied depends on the modality:

- **Widefield**: the irradiance is uniform, so the emission rates are reduced
  by \(1 / (1 + s)\).
- **Confocal**: the irradiance varies across the focused spot, so saturation
  is applied at each position of the excitation PSF. The peak saturates first,
  which broadens the effective PSF. Here `power` is the irradiance at the
  center of the spot, and `filtered_emission_rates()` gives the (saturated)
  rates there.

Saturation requires `lifetime_ns` on the fluorophore. Fluorophores from FPbase
usually have one.

!!! info "Not modeled"

    Triplet and other dark states, stimulated emission, the orientation of
    the absorption dipole relative to polarized excitation, and pulsed
    excitation. With pulsed lasers or with dark-state buildup, real
    fluorophores saturate at lower average irradiance than this model gives.

## From rates to images

The emission rate of each fluorophore is per molecule. The
[next stage](modalities.md) multiplies it by the number of fluorophores in each
voxel and convolves with the PSF. The widefield PSF is normalized so that each
plane sums to about 1: a camera focused at any depth records all the light that
the objective collected from a point source. The confocal PSF is scaled by the
relative excitation intensity at each position and by the fraction of
collected light that passes the pinhole.
