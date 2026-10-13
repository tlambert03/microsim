from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import xarray as xr

from microsim.schema.dimensions import Axis

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    absorption_rate : xr.DataArray
        Photons absorbed per second by one fluorophore, at each excitation wavelength,
        from all light sources that are on while this channel is acquired (i.e. those
        of every channel in its track).
    detected_fraction : xr.DataArray
        Photons detected per photon absorbed, at each emission wavelength: quantum
        yield x emission spectrum x emission filters x detector QE x the fraction of
        emitted light collected by the objective.
    track : int
        Index of the track this channel belongs to: the group of channels acquired
        at the same time (see `Simulation.tracks`).
    """

    channel: OpticalConfig
    fluorophore: Fluorophore
    absorption_rate: xr.DataArray  # (W,)
    detected_fraction: xr.DataArray  # (W,)
    track: int = 0

    @classmethod
    def create(
        cls,
        channel: OpticalConfig,
        fluorophore: Fluorophore,
        objective: ObjectiveLens,
        detector: Detector | None = None,
        *,
        absorption_rate: xr.DataArray | None = None,
        track: int = 0,
    ) -> ChannelFluorescence:
        """Describe `fluorophore` in `channel`.

        `absorption_rate` is the fluorophore's absorption from all light sources of
        the channel's track (see `track_absorption_rate`).  By default, only from
        `channel` itself.
        """
        if absorption_rate is None:
            absorption_rate = track_absorption_rate([channel], fluorophore)
        qe = detector.qe if detector else None
        detected = channel.detected_fraction(fluorophore, detector_qe=qe)
        # fraction of isotropic emission entering the objective. PSFs are normalized
        # to collected light, so this is not double-counted (see `detection_psf`)
        detected = detected * objective.collection_efficiency
        return cls(channel, fluorophore, absorption_rate, detected, track)

    @property
    def total_absorption_rate(self) -> float:
        """Photons absorbed per second by one fluorophore (all wavelengths)."""
        return float(self.absorption_rate.sum())

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


def track_light_sources(track: Sequence[OpticalConfig]) -> list[OpticalConfig]:
    """Return the channels whose light sources excite `track`, one per light source.

    A light source used by several channels of a track (e.g. two detection windows
    behind one laser) is on once, so those channels must agree on its power and on
    the excitation-side filters (e.g. the dichroic).
    """
    sources: dict[tuple, OpticalConfig] = {}
    light_owner: dict[str, OpticalConfig] = {}
    for oc in track:
        # channels without lights are illuminated through their filters (a lamp)
        key = tuple(sorted(oc._light_keys())) if oc.lights else oc._excitation_key()
        first = sources.setdefault(key, oc)
        if first is not oc:
            if first._excitation_key() != oc._excitation_key():
                raise ValueError(
                    f"Channels {first.name!r} and {oc.name!r} are acquired together "
                    "(in the same track) with the same light source, so they must "
                    "use the same power and excitation filters (e.g. dichroic)."
                )
            continue
        for light in oc._light_keys():
            if (other := light_owner.setdefault(light, oc)) is not oc:
                raise ValueError(
                    f"Channels {other.name!r} and {oc.name!r} are acquired together "
                    "(in the same track) and share some, but not all, light sources. "
                    "Give each light source to one channel (or the same set of light "
                    "sources to both)."
                )
    return list(sources.values())


def track_absorption_rate(
    track: Sequence[OpticalConfig], fluorophore: Fluorophore
) -> xr.DataArray:
    """Photons absorbed per second by `fluorophore`, from all light sources of `track`.

    Returns one value per excitation wavelength (dimension `W`).
    """
    rates = [
        oc.absorption_rate(fluorophore).squeeze([Axis.F, Axis.C], drop=True)
        for oc in track_light_sources(track)
    ]
    if len(rates) == 1:
        return rates[0]
    return xr.concat(rates, dim="source", join="outer", fill_value=0).sum("source")
