from typing import Literal

from microsim._data_array import DataArray
from microsim.interval_creation import WavelengthSpace
from microsim.psf import make_psf
from microsim.schema._base_model import SimBaseModel
from microsim.schema.backend import NumpyAPI
from microsim.schema.lens import ObjectiveLens
from microsim.schema.optical_config import OpticalConfig
from microsim.schema.settings import Settings


class Widefield(SimBaseModel):
    type: Literal["widefield"] = "widefield"

    def render(
        self,
        truth: WavelengthSpace,
        channel: OpticalConfig,
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI | None = None,
    ) -> DataArray:
        xp = NumpyAPI.create(xp)

        # TODO: @ashesh: Here, we can iterate over the psfs.
        em_psf = make_psf(
            space=truth.attrs["space"],
            channel=channel,
            objective=objective_lens,
            max_au_relative=settings.max_psf_radius_aus,
            xp=xp,
        )

        img = xp.fftconvolve(truth.data, em_psf, mode="same")
        return DataArray(img, coords=truth.coords, attrs=truth.attrs)
