from typing import Annotated, Any, Literal

import pint
from annotated_types import Ge
from pint import Quantity

from microsim._data_array import ArrayProtocol, DataArray, xrDataArray
from microsim.psf import make_psf
from microsim.schema._base_model import SimBaseModel
from microsim.schema.backend import NumpyAPI
from microsim.schema.dimensions import Axis
from microsim.schema.lens import ObjectiveLens
from microsim.schema.optical_config import OpticalConfig
from microsim.schema.settings import Settings
from microsim.schema.space import SpaceProtocol


class _PSFModality(SimBaseModel):
    def psf(
        self,
        space: SpaceProtocol,
        channel: OpticalConfig,
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI,
        em_wvl: Quantity | None = None,
    ) -> ArrayProtocol:
        # default implementation is a widefield PSF
        return make_psf(
            space=space,
            channel=channel,
            objective=objective_lens,
            max_au_relative=settings.max_psf_radius_aus,
            xp=xp,
            em_wvl=em_wvl,
        )

    def render(
        self,
        truth: xrDataArray,
        channel: OpticalConfig,
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI,
    ) -> xrDataArray:
        convolved: Any = 0
        ureg = pint.application_registry.get()  # type: ignore
        for fluor_idx in range(truth.sizes[Axis.F]):
            convolved_fluor: Any = 0
            for bin_idx in range(truth.sizes[Axis.W]):
                binned_flux = truth.isel({Axis.W: bin_idx, Axis.F: fluor_idx})
                if xp.isnan(xp.sum(binned_flux.data)):
                    # NOTE: there can be bins for which there is no data in one of the
                    #  fluorophores
                    continue
                em_wvl = binned_flux[Axis.W].values.item().mid * ureg.nm
                psf = self.psf(
                    truth.attrs["space"],
                    channel,
                    objective_lens,
                    settings,
                    xp,
                    em_wvl=em_wvl,
                )
                convolved_fluor += xp.fftconvolve(
                    binned_flux.isel({Axis.C: 0}), psf, mode="same"
                )
            convolved += convolved_fluor

        return DataArray(
            convolved[None],
            dims=[Axis.C, Axis.Z, Axis.Y, Axis.X],
            coords={
                Axis.C: [channel],
                Axis.Z: truth.coords[Axis.Z],
                Axis.Y: truth.coords[Axis.Y],
                Axis.X: truth.coords[Axis.X],
            },
            attrs=truth.attrs,
        )


class Confocal(_PSFModality):
    type: Literal["confocal"] = "confocal"
    pinhole_au: Annotated[float, Ge(0)] = 1

    def psf(
        self,
        space: SpaceProtocol,
        channel: OpticalConfig,
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI,
        em_wvl: Quantity | None = None,
    ) -> ArrayProtocol:
        return make_psf(
            space=space,
            channel=channel,
            objective=objective_lens,
            pinhole_au=self.pinhole_au,
            max_au_relative=settings.max_psf_radius_aus,
            xp=xp,
            em_wvl=em_wvl,
        )


class Widefield(_PSFModality):
    type: Literal["widefield"] = "widefield"
