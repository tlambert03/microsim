from typing import Literal

from microsim._data_array import DataArray, xrDataArray
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
        truth: xrDataArray,
        channel: OpticalConfig,
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI | None = None,
    ) -> xrDataArray:
        xp = NumpyAPI.create(xp)

        em_psf = make_psf(
            space=truth.attrs["space"],
            channel=channel,
            objective=objective_lens,
            max_au_relative=settings.max_psf_radius_aus,
            xp=xp,
        )
        if truth.ndim > 3:
            if truth.ndim > 4:
                raise ValueError("truth data must be 3D or 4D")
            # do 3d convolution for each item other axes
            images = [xp.fftconvolve(t.data, em_psf, mode="same") for t in truth]
            img = xp.stack(images)
        else:
            img = xp.fftconvolve(truth.data, em_psf, mode="same")

        return DataArray(img, dims=truth.dims, coords=truth.coords, attrs=truth.attrs)
