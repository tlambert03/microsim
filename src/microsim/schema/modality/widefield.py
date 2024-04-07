from typing import TYPE_CHECKING, Literal, cast

import xarray
from pydantic import BaseModel

from microsim.schema.channel import Channel
from microsim.schema.lens import ObjectiveLens
from microsim.schema.settings import NumpyAPI

if TYPE_CHECKING:
    from microsim.schema.space import Space


class Widefield(BaseModel):
    type: Literal["widefield"] = "widefield"

    def render(
        self,
        truth: xarray.DataArray,
        channel: Channel,
        objective_lens: ObjectiveLens,
        xp: NumpyAPI | None = None,
    ) -> xarray.DataArray:
        from psfmodels import vectorial_psf_centered

        # FIXME, this is probably derivable from truth.coords
        truth_space = cast("Space", truth.attrs["space"])
        em_psf = vectorial_psf_centered(
            wvl=channel.emission.wavelength * 1e-3,
            nz=truth_space.shape[-3] + 1,
            nx=truth_space.shape[-1] + 1,
            dz=truth_space.scale[-3],
            dxy=truth_space.scale[-1],
            params={"NA": objective_lens.numerical_aperture},
        )

        em_psf = xp.asarray(em_psf)
        return xp.fftconvolve(truth.data, em_psf, mode="same")
