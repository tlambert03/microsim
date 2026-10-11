from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from microsim.schema.dimensions import Axis

if TYPE_CHECKING:
    import xarray as xr

    from microsim.schema.detectors import Detector
    from microsim.schema.lens import ObjectiveLens
    from microsim.schema.sample.fluorophore import Fluorophore

    from .config import OpticalConfig


@dataclass(frozen=True, eq=False)
class ChannelFluorescence:
    """How bright one fluorophore is in one channel, and at which wavelengths.

    This has no spatial information: it describes a single fluorophore at the
    *reference* illumination intensity (`OpticalConfig.power`), which is the intensity
    everywhere in a widefield field of view, or at the center of a focused spot.  It
    is unsaturated: modalities apply saturation, since it depends on where the
    fluorophore sits in the illumination.

    Attributes
    ----------
    channel : OpticalConfig
        The channel (light sources, filters).
    fluorophore : Fluorophore
        The fluorophore.
    absorption_spectrum : xr.DataArray
        Photons absorbed per second by one fluorophore, at each excitation wavelength.
    detected_fraction : xr.DataArray
        Photons detected per photon absorbed, at each emission wavelength: quantum
        yield x emission spectrum x emission filters x detector QE x the fraction of
        emitted light collected by the objective.
    """

    channel: OpticalConfig
    fluorophore: Fluorophore
    absorption_spectrum: xr.DataArray  # (W,)
    detected_fraction: xr.DataArray  # (W,)

    @classmethod
    def create(
        cls,
        channel: OpticalConfig,
        fluorophore: Fluorophore,
        objective: ObjectiveLens,
        detector: Detector | None = None,
    ) -> ChannelFluorescence:
        qe = detector.qe if detector else None
        absorption = channel.absorption_rate(fluorophore)
        absorption = absorption.squeeze([Axis.F, Axis.C], drop=True)
        detected = channel.detected_fraction(fluorophore, detector_qe=qe)
        # fraction of isotropic emission entering the objective. PSFs are normalized
        # to collected light, so this is not double-counted (see `detection_psf`)
        detected = detected * objective.collection_efficiency
        return cls(channel, fluorophore, absorption, detected)

    @property
    def total_absorption_rate(self) -> float:
        """Photons absorbed per second by one fluorophore (all wavelengths)."""
        return float(self.absorption_spectrum.sum())

    @property
    def saturation(self) -> float:
        """How strongly the fluorophore is saturated at the reference intensity.

        This is the absorption rate times the excited-state lifetime (0 if the
        lifetime is unknown).  Emission is reduced by a factor of `1 + saturation`
        relative to a linear response: at 1, a fluorophore emits half as much as it
        would without saturation.
        """
        if not (lifetime_ns := self.fluorophore.lifetime_ns):
            return 0.0
        return self.total_absorption_rate * lifetime_ns * 1e-9

    def detected_rate(self, *, saturate: bool = True) -> float:
        """Photons detected per second from one fluorophore (all wavelengths).

        If `saturate` is True, saturation at the reference intensity is applied.
        """
        rate = self.total_absorption_rate * float(self.detected_fraction.sum())
        return rate / (1 + self.saturation) if saturate else rate

    def detected_spectrum(self, *, saturate: bool = True) -> xr.DataArray:
        """Photons detected per second from one fluorophore, at each wavelength.

        If `saturate` is True, saturation at the reference intensity is applied.
        """
        rate = self.total_absorption_rate
        if saturate:
            rate /= 1 + self.saturation
        out = self.detected_fraction * rate
        out = out.expand_dims({Axis.F: [self.fluorophore], Axis.C: [self.channel]})
        out.name = "filtered_emission_rate"
        out.attrs.update(long_name="Filtered Emission rate", units="photons/s")
        return out
