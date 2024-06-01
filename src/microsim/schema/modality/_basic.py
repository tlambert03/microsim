from typing import Annotated, Literal

from annotated_types import Ge

from microsim._data_array import ArrayProtocol, DataArray
from microsim.psf import make_psf
from microsim.schema._base_model import SimBaseModel
from microsim.schema.backend import NumpyAPI
from microsim.schema.lens import ObjectiveLens
from microsim.schema.optical_config import OpticalConfig
from microsim.schema.settings import Settings
from microsim.schema.space import SpaceProtocol


class _PSFBasedModality(SimBaseModel):
    """Basic class for modalities that are based on a PSF."""

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
        truth: DataArray,
        channel: OpticalConfig,
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI | None = None,
    ) -> DataArray:
        xp = NumpyAPI.create(xp)

        # convert label dimension to photon flux
        psf = self.psf(truth.attrs["space"], channel, objective_lens, settings, xp)

        xa = truth.to_xarray()
        for i in range(xa.sizes["L"]):
            print("convolving layer", i)
            xa[{"L": i}] = xp.fftconvolve(xa.isel(L=i).data, psf, mode="same")

        return DataArray.from_xarray(xa)


class Confocal(_PSFBasedModality):
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


class Widefield(_PSFBasedModality):
    type: Literal["widefield"] = "widefield"
