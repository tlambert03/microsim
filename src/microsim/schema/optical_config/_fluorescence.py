from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from microsim.schema.dimensions import Axis

if TYPE_CHECKING:
    import xarray as xr

    from microsim.schema.detectors import Detector
    from microsim.schema.lens import ObjectiveLens
    from microsim.schema.sample.fluorophore import Fluorophore
    from microsim.schema.spectrum import Spectrum

    from .config import OpticalConfig


@dataclass(frozen=True, eq=False)
class ChannelFluorescence:
    """How bright one fluorophore is in one channel, and at which wavelengths.

    This has no spatial information: it describes a single fluorophore at the
    *reference* illumination intensity (`OpticalConfig.power`), which is the intensity
    everywhere in a widefield field of view, or at the center of a focused spot.  The
    rates are given with or without saturation at that intensity: modalities that
    don't illuminate evenly (e.g. confocal) apply saturation locally instead.

    Modalities receive one of these for each channel and fluorophore in `render()`.

    Attributes
    ----------
    channel : OpticalConfig
        The channel (light sources, filters).
    fluorophore : Fluorophore
        The fluorophore.
    absorption_rate : xr.DataArray
        Photons absorbed per second by one fluorophore, at each excitation wavelength.
    detected_fraction : xr.DataArray
        Photons detected per photon absorbed, at each emission wavelength: quantum
        yield x emission spectrum x emission filters x detector QE x the fraction of
        emitted light collected by the objective.
    """

    channel: OpticalConfig
    fluorophore: Fluorophore
    absorption_rate: xr.DataArray  # (W,)
    detected_fraction: xr.DataArray  # (W,)

    @classmethod
    def create(
        cls,
        channel: OpticalConfig,
        fluorophore: Fluorophore,
        objective: ObjectiveLens,
        detector: Detector | None = None,
    ) -> ChannelFluorescence:
        """Describe `fluorophore` in `channel`."""
        qe = detector.qe if detector else None
        fl = cls.from_channel(channel, fluorophore, detector_qe=qe)
        # fraction of isotropic emission entering the objective. PSFs are normalized
        # to collected light, so this is not double-counted (see `detection_psf`)
        detected = fl.detected_fraction * objective.collection_efficiency
        return replace(fl, detected_fraction=detected)

    @classmethod
    def from_channel(
        cls,
        channel: OpticalConfig,
        fluorophore: Fluorophore,
        detector_qe: float | Spectrum | None = None,
    ) -> ChannelFluorescence:
        """Describe `fluorophore` in `channel`, ignoring the objective.

        Unlike `create`, the detected fraction doesn't include the fraction of
        emitted light collected by the objective.
        """
        absorption_rate = channel.absorption_rate(fluorophore).squeeze(
            [Axis.F, Axis.C], drop=True
        )
        detected = channel.detected_fraction(fluorophore, detector_qe)
        return cls(channel, fluorophore, absorption_rate, detected)

    @property
    def saturation(self) -> float:
        """How strongly the fluorophore is saturated at the reference intensity.

        This is the (unsaturated) absorption rate times the excited-state lifetime
        (0 if the lifetime is unknown): `I / I_sat` for a two-level system.  Emission
        is reduced by a factor of `1 + saturation` relative to a linear response: at
        1, a fluorophore emits half as much as it would without saturation.
        """
        # Steady state of a two-level system, under continuous illumination.
        # Known omissions (not modeled):
        # - triplet and other dark/intermediate states (S0 <-> S1 only)
        # - stimulated emission
        # - orientation: with polarized excitation, k varies per molecule and
        #   <k/(1+k*tau)> != <k>/(1+<k>*tau); using mean k over-saturates
        # - pulsed excitation: assumes CW (steady state), not pulses << lifetime
        if not (lifetime_ns := self.fluorophore.lifetime_ns):
            return 0.0
        return self.total_absorption_rate(saturate=False) * lifetime_ns * 1e-9

    def total_absorption_rate(self, *, saturate: bool = True) -> float:
        """Photons absorbed per second by one fluorophore (all wavelengths).

        If `saturate` is True, saturation at the reference intensity is applied: a
        saturated fluorophore spends more time in the excited state, where it can't
        absorb.
        """
        rate = float(self.absorption_rate.sum())
        return rate / (1 + self.saturation) if saturate else rate

    def detected_rate(self, *, saturate: bool = True) -> float:
        """Photons detected per second from one fluorophore (all wavelengths).

        If `saturate` is True, saturation at the reference intensity is applied.
        """
        absorbed = self.total_absorption_rate(saturate=saturate)
        return absorbed * float(self.detected_fraction.sum())

    def detected_spectrum(self, *, saturate: bool = True) -> xr.DataArray:
        """Photons detected per second from one fluorophore, at each wavelength.

        If `saturate` is True, saturation at the reference intensity is applied.
        """
        out = self.detected_fraction * self.total_absorption_rate(saturate=saturate)
        out = out.expand_dims({Axis.F: [self.fluorophore], Axis.C: [self.channel]})
        out.name = "filtered_emission_rate"
        out.attrs.update(long_name="Filtered Emission rate", units="photons/s")
        return out
