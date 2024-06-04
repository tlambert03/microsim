from typing import TYPE_CHECKING, Annotated, Literal

from annotated_types import Ge

from microsim._data_array import DataArray
from microsim.psf import make_psf
from microsim.schema._base_model import SimBaseModel
from microsim.schema.backend import NumpyAPI
from microsim.schema.lens import ObjectiveLens
from microsim.schema.optical_config import OpticalConfig
from microsim.schema.settings import Settings

if TYPE_CHECKING:
    import xarray as xr


class Confocal(SimBaseModel):
    type: Literal["confocal"] = "confocal"
    pinhole_au: Annotated[float, Ge(0)] = 1

    def render(
        self,
        truth: "xr.DataArray",
        channel: OpticalConfig,
        objective_lens: ObjectiveLens,
        settings: Settings,
        xp: NumpyAPI | None = None,
    ) -> "xr.DataArray":
        xp = NumpyAPI.create(xp)

        psf = make_psf(
            space=truth.attrs["space"],
            channel=channel,
            objective=objective_lens,
            pinhole_au=self.pinhole_au,
            max_au_relative=settings.max_psf_radius_aus,
            xp=xp,
        )

        if truth.ndim > 3:
            if truth.ndim > 4:
                raise ValueError("truth data must be 3D or 4D")
            # do 3d convolution for each item other axes
            images = [xp.fftconvolve(t.data, psf, mode="same") for t in truth]
            img = xp.stack(images)
        else:
            img = xp.fftconvolve(truth.data, psf, mode="same")

        return DataArray(img, dims=truth.dims, coords=truth.coords, attrs=truth.attrs)
