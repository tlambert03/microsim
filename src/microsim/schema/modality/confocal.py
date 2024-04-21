from typing import Annotated, Literal

from annotated_types import Ge

from microsim._data_array import DataArray
from microsim.psf import make_psf
from microsim.schema._base_model import SimBaseModel
from microsim.schema.backend import NumpyAPI
from microsim.schema.lens import ObjectiveLens
from microsim.schema.optical_config import OpticalConfig
from microsim.schema.settings import Settings


class Confocal(SimBaseModel):
    type: Literal["confocal"] = "confocal"
    pinhole_au: Annotated[float, Ge(0)] = 1

    def render(
        self,
        truth: DataArray,
        channel: OpticalConfig,
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI | None = None,
    ) -> DataArray:
        xp = NumpyAPI.create(xp)

        psf = make_psf(
            space=truth.attrs["space"],
            channel=channel,
            objective=objective_lens,
            pinhole_au=self.pinhole_au,
            max_au_relative=settings.max_psf_radius_aus,
            xp=xp,
        )

        img = xp.fftconvolve(truth.data, psf, mode="same")
        return DataArray(img, coords=truth.coords, attrs=truth.attrs)
