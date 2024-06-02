from typing import Annotated, Literal

from annotated_types import Ge

from microsim._data_array import DataArray
from microsim.interval_creation import WavelengthSpace
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
        truth: WavelengthSpace,
        channel: OpticalConfig,
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI | None = None,
    ) -> DataArray:
        xp = NumpyAPI.create(xp)

        bins = truth.wavelength_bins
        psf = make_psf(
            space=truth.attrs["space"],
            emission_wavelength_bins=bins,
            channel=channel,
            objective=objective_lens,
            pinhole_au=self.pinhole_au,
            max_au_relative=settings.max_psf_radius_aus,
            xp=xp,
        )

        img_list = [
            xp.fftconvolve(truth.data[wv_idx], psf[wv_idx], mode="same")
            for wv_idx in range(len(bins))
        ]

        img = 0
        for i in range(len(bins)):
            img += img_list[i]

        return DataArray(img, coords=truth.coords, attrs=truth.attrs)
