from typing import Annotated

import numpy as np
import numpy.typing as npt
from annotated_types import Ge, Interval
from pydantic import Field
from scipy import stats

from microsim._data_array import DataArray, xrDataArray
from microsim.schema._base_model import SimBaseModel
from microsim.schema.backend import NumpyAPI

PositiveFloat = Annotated[float, Ge(0)]
PositiveInt = Annotated[int, Ge(0)]


class Camera(SimBaseModel):
    read_noise: PositiveFloat = 6  # as function of readout rate?
    qe: Annotated[float, Interval(ge=0, le=1)] = 1  # TODO: spectrum
    full_well: int = 18_000
    # TODO: serial register fullwell?
    dark_current: PositiveFloat = Field(0.001, description="e/pix/sec")
    clock_induced_charge: PositiveFloat = Field(0, description="e/pix/sec")
    bit_depth: PositiveInt = 12
    offset: int = 100
    gain: PositiveFloat = 1
    readout_rate: PositiveFloat = Field(1, description="MHz")
    # npixels_h: int = 1000
    # npixels_v: int = 1000
    # photodiode_size: float = 6.5

    # binning?  or keep in simulate

    @property
    def dynamic_range(self) -> float:
        return self.full_well / self.read_noise

    def apply_em_gain(self, electron_image: npt.NDArray) -> npt.NDArray:
        return electron_image

    def render(
        self,
        image: xrDataArray,
        exposure_ms: float = 100,
        binning: int = 1,
        add_poisson: bool = True,
        xp: NumpyAPI | None = None,
    ) -> xrDataArray:
        """Simulate imaging process.

        Parameters
        ----------
        image : DataArray
            array where each element represents photons / second
        exposure_ms : float, optional
            Exposure time in milliseconds, by default 100
        binning : int, optional
            Binning to apply, by default 1
        add_poisson : bool, optional
            Whether to add poisson noise, by default True
        xp: NumpyAPI | None
            Numpy API backend
        """
        from microsim.schema.detectors import simulate_camera

        new_data = simulate_camera(
            camera=self,
            image=image,
            exposure_ms=exposure_ms,
            binning=binning,
            add_poisson=add_poisson,
            xp=xp,
        )
        return DataArray(
            new_data, dims=image.dims, coords=image.coords, attrs=image.attrs
        )

    @property
    def adc_gain(self) -> float:
        # assume ADC has been set such that voltage at FWC matches max bit depth
        return self.full_well / self.max_intensity

    @property
    def max_intensity(self) -> int:
        return int(2**self.bit_depth - 1)

    def quantize_electrons(
        self, total_electrons: npt.NDArray, xp: NumpyAPI
    ) -> npt.NDArray:
        voltage = xp.norm_rvs(total_electrons, self.read_noise) * self.gain
        return xp.round((voltage / self.adc_gain) + self.offset)  # type: ignore


class CameraCCD(Camera): ...


class CameraEMCCD(Camera):
    em_full_well: int
    em_gain: float

    def apply_em_gain(self, electron_image: npt.NDArray) -> npt.NDArray:
        # FIXME: is there a more elegant way to deal with gamma rvs with shape = 0?
        ind_zero = electron_image <= 0
        electron_image[ind_zero] += 1
        # gamma shape is input electrons / scale is EM gain
        electron_image = stats.gamma.rvs(
            electron_image.astype(float), scale=self.em_gain
        )
        electron_image[ind_zero] = 0
        electron_image = np.round(electron_image).astype(int)
        # cap to EM full-well-capacity
        return np.minimum(electron_image, self.em_full_well)


class CameraCMOS(Camera): ...


ICX285 = CameraCCD(
    # photodiode_size=6.45,
    qe=0.70,
    gain=1,
    full_well=18000,
    dark_current=0.0005,
    clock_induced_charge=1,
    read_noise=6,
    readout_rate=14,
    bit_depth=12,
    offset=100,
    # npixels_h=1344,
    # npixels_v=1024,
)
