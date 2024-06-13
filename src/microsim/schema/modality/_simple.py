from typing import Annotated, Literal

from annotated_types import Ge

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
    ) -> ArrayProtocol:
        # default implementation is a widefield PSF
        return make_psf(
            space=space,
            channel=channel,
            objective=objective_lens,
            max_au_relative=settings.max_psf_radius_aus,
            xp=xp,
        )

    def render(
        self,
        truth: xrDataArray,
        channel: OpticalConfig,
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI | None = None,
    ) -> xrDataArray:
        if truth.ndim > 4 or Axis.F not in truth.dims:
            raise NotImplementedError(
                "At this stage, we only support rendering 3D or 4D data with a "
                "fluorophore dimension."
            )

        xp = NumpyAPI.create(xp)
        # convert label dimension to photon flux
        psf = self.psf(truth.attrs["space"], channel, objective_lens, settings, xp)
        convolved = [
            xp.fftconvolve(truth.isel({Axis.F: f}).data, psf, mode="same")
            for f in range(truth.sizes[Axis.F])
        ]
        img = xp.stack(convolved)

        return DataArray(img, dims=truth.dims, coords=truth.coords, attrs=truth.attrs)


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
    ) -> ArrayProtocol:
        return make_psf(
            space=space,
            channel=channel,
            objective=objective_lens,
            pinhole_au=self.pinhole_au,
            max_au_relative=settings.max_psf_radius_aus,
            xp=xp,
        )


class Widefield(_PSFModality):
    type: Literal["widefield"] = "widefield"
