from typing import TYPE_CHECKING, Annotated, Literal, cast

from annotated_types import Ge
from pydantic import BaseModel

from microsim._data_array import DataArray
from microsim.schema.backend import NumpyAPI
from microsim.schema.lens import ObjectiveLens
from microsim.schema.optical_config import OpticalConfig

if TYPE_CHECKING:
    from microsim.schema.space import Space


class Confocal(BaseModel):
    type: Literal["confocal"] = "confocal"
    pinhole_au: Annotated[float, Ge(0)] = 1

    def render(
        self,
        truth: DataArray,
        channel: OpticalConfig,
        objective_lens: ObjectiveLens,
        xp: NumpyAPI | None = None,
    ) -> DataArray:
        from microsim.util import make_confocal_psf

        xp = xp or NumpyAPI()

        # FIXME, this is probably derivable from truth.coords
        truth_space = cast("Space", truth.attrs["space"])
        psf = make_confocal_psf(
            ex_wvl_um=channel.excitation.bandcenter * 1e-3,
            em_wvl_um=channel.emission.bandcenter * 1e-3,
            pinhole_au=self.pinhole_au,
            nz=truth_space.shape[-3] + 1,
            nx=truth_space.shape[-1] + 1,
            dz=truth_space.scale[-3],
            dxy=truth_space.scale[-1],
            params={"NA": objective_lens.numerical_aperture},
            xp=xp,
        )
        psf = xp.asarray(psf)
        img = xp.fftconvolve(truth.data, psf, mode="same")
        return DataArray(img, coords=truth.coords, attrs=truth.attrs)
