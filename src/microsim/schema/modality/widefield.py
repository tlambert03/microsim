from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel

from microsim._data_array import DataArray
from microsim.psf import vectorial_psf_centered
from microsim.schema.backend import NumpyAPI
from microsim.schema.lens import ObjectiveLens
from microsim.schema.optical_config import OpticalConfig

if TYPE_CHECKING:
    from microsim.schema.space import Space


class Widefield(BaseModel):
    type: Literal["widefield"] = "widefield"

    def render(
        self,
        truth: DataArray,
        channel: OpticalConfig,
        objective_lens: ObjectiveLens,
        xp: NumpyAPI | None = None,
    ) -> DataArray:
        xp = NumpyAPI.create(xp)

        # FIXME, this is probably derivable from truth.coords
        truth_space = cast("Space", truth.attrs["space"])
        em_psf = vectorial_psf_centered(
            wvl=channel.emission.bandcenter * 1e-3,
            nz=truth_space.shape[-3] + 1,
            nx=truth_space.shape[-1] + 1,
            dz=truth_space.scale[-3],
            dxy=truth_space.scale[-1],
            objective_params={"numerical_aperture": objective_lens.numerical_aperture},
        )

        em_psf = xp.asarray(em_psf)
        img = xp.fftconvolve(truth.data, em_psf, mode="same")
        return DataArray(img, coords=truth.coords, attrs=truth.attrs)
