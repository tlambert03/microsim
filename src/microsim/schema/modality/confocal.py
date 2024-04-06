from typing import TYPE_CHECKING, Annotated, Literal, cast

import xarray
from annotated_types import Ge
from pydantic import BaseModel

from microsim.schema.channel import Channel
from microsim.schema.lens import ObjectiveLens

if TYPE_CHECKING:
    from microsim.schema.space import Space



class Confocal(BaseModel):
    type: Literal["confocal"] = "confocal"
    pinhole: Annotated[float, Ge(0)] = 1

    def render(
        self, truth: xarray.DataArray, channel: Channel, objective_lens: ObjectiveLens
    ):
        from scipy import signal

        from microsim.util import make_confocal_psf

        # FIXME, this is probably derivable from truth.coords
        truth_space = cast("Space", truth.attrs["space"])
        psf = make_confocal_psf(
            ex_wvl_um=channel.excitation.wavelength * 1e-3,
            em_wvl_um=channel.emission.wavelength * 1e-3,
            pinhole_au=self.pinhole,
            nz=truth_space.shape[-3] + 1,
            nx=truth_space.shape[-1] + 1,
            dz=truth_space.scale[-3],
            dxy=truth_space.scale[-1],
            params={"NA": objective_lens.numerical_aperture},
        )

        img = signal.fftconvolve(truth, psf, mode="same")
        return img

