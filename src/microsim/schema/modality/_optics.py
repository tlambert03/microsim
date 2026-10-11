"""How a modality illuminates the sample, and how it detects the emitted light.

Every modality in this package forms its image the same way: the signal from one
fluorophore, at some position relative to the focus, is

    (how strongly it is excited there) x (how much of its emission is detected)

i.e. the excitation spot times the detection PSF: the *system PSF*.  The
`Illumination` classes describe the first factor and the `Detection` classes the
second.  A modality picks one of each.  (In widefield, excitation is uniform, so the
system PSF is just the detection PSF.)

Everything here is internal (not part of the user-facing schema): modalities build
these objects themselves.  They are frozen dataclasses, so they can be used directly
in PSF cache keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from microsim.psf import detection_psf, excitation_spot

if TYPE_CHECKING:
    from microsim._data_array import ArrayProtocol
    from microsim.psf import PSFGrid
    from microsim.schema.backend import NumpyAPI
    from microsim.schema.lens import ObjectiveLens

# (wavelength_nm, weight) pairs: a spectrum discretized into a few wavelengths
SpectralLines = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class WidefieldIllumination:
    """The whole field of view is evenly illuminated."""

    def spot(
        self,
        excitation: SpectralLines,
        grid: PSFGrid,
        objective: ObjectiveLens,
        xp: NumpyAPI,
    ) -> float:
        """Relative excitation intensity: the same (1) everywhere."""
        return 1.0


@dataclass(frozen=True)
class FocusedIllumination:
    """Excitation light is focused to a diffraction-limited spot (point scanning)."""

    def spot(
        self,
        excitation: SpectralLines,
        grid: PSFGrid,
        objective: ObjectiveLens,
        xp: NumpyAPI,
    ) -> ArrayProtocol:
        """Relative excitation intensity at each position (1 at the focus).

        With several excitation wavelengths (e.g. two lasers, or a broadband source),
        each wavelength's spot is weighted by how much of the fluorophore's total
        absorption it accounts for (the `excitation` weights).
        """
        total = sum(weight for _, weight in excitation)
        spot: Any = 0
        for wvl_nm, weight in excitation:
            spot = spot + excitation_spot(grid, wvl_nm, objective, xp) * (
                weight / total
            )
        return spot  # type: ignore [no-any-return]


@dataclass(frozen=True)
class CameraDetection:
    """The sample is imaged onto a camera."""

    # each camera pixel collects all of the light falling on its area
    sums_over_pixel_area: ClassVar[bool] = True

    def psf(
        self, wvl_nm: float, grid: PSFGrid, objective: ObjectiveLens, xp: NumpyAPI
    ) -> ArrayProtocol:
        """Fraction of the light collected from a point source, on each pixel."""
        return detection_psf(grid, wvl_nm, objective, xp)

    def psf_radius_aus(
        self, depth_um: float, wvl_um: float, objective: ObjectiveLens
    ) -> float:
        """Lateral PSF radius (in Airy units) for `max_psf_radius_aus="auto"`.

        Calibrated to ~1% RMSE vs. an untruncated PSF: ~14 AU covers the tails of the
        in-focus Airy pattern, plus half the radius of the cone of out-of-focus light
        at the axial edge of the PSF (half the volume depth).
        """
        airy_um = 0.61 * wvl_um / objective.numerical_aperture
        cone_aus = depth_um / 2 * np.tan(objective.half_angle) / airy_um
        return float(14 + cone_aus / 2)


@dataclass(frozen=True)
class PinholeDetection:
    """A single detector behind a pinhole, which records one scan position at a time.

    `radius_um` is the radius of the pinhole, projected into the sample (i.e. its
    physical radius divided by the magnification).  It is the same for all
    fluorophores and wavelengths in a channel.
    """

    radius_um: float

    # each pixel is the signal recorded at one scan position
    sums_over_pixel_area: ClassVar[bool] = False

    def psf(
        self, wvl_nm: float, grid: PSFGrid, objective: ObjectiveLens, xp: NumpyAPI
    ) -> ArrayProtocol:
        """Fraction of collected light passing the pinhole, at each source position."""
        return detection_psf(grid, wvl_nm, objective, xp, self.radius_um)

    def psf_radius_aus(
        self, depth_um: float, wvl_um: float, objective: ObjectiveLens
    ) -> float:
        # the pinhole rejects out-of-focus light, so a fixed radius suffices
        # (<0.3% RMSE vs. an untruncated PSF, for volumes up to ~5 um deep)
        return 8


Illumination = WidefieldIllumination | FocusedIllumination
Detection = CameraDetection | PinholeDetection
