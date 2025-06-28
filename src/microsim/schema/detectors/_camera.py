from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
import numpy.typing as npt
from annotated_types import Ge, Interval
from pydantic import Field

from microsim._data_array import DataArray, xrDataArray
from microsim.schema._base_model import SimBaseModel
from microsim.schema.backend import NumpyAPI
from microsim.schema.spectrum import Spectrum
from microsim.util import bin_window

if TYPE_CHECKING:
    from microsim._data_array import ArrayProtocol

PositiveFloat = Annotated[float, Ge(0)]
PositiveInt = Annotated[int, Ge(0)]


class _Camera(SimBaseModel):
    """Base Camera model.

    Attributes
    ----------
    camera_type : str
        Type of camera, for discriminated union.
    read_noise : float
        Read noise in electrons.
    qe : float
        Quantum efficiency, from 0-1. If a float, it is assumed to be constant across
        all wavelengths. If a Spectrum, it is assumed to be a function of wavelength.
    full_well : int
        Full well capacity in electrons.
    serial_reg_full_well : int, optional
        Serial register full well capacity in electrons.
    dark_current : float
        Dark current in electrons per pixel per second.
    clock_induced_charge : float
        Clock induced charge in electrons per pixel per second.
    bit_depth : int
        Bit depth of the camera.
    offset : int
        Offset of the camera, in gray values.
    gain : float
        Gain of the camera. At the default gain of 1, the camera will reach full well
        capacity at the maximum intensity value of the ADC bit depth.
    name : str
        A descriptive name for the camera.  Not used internally.
    """

    camera_type: str = "generic"
    read_noise: PositiveFloat = 6  # TODO: accept map of readout rate -> noise?
    qe: Annotated[float, Interval(ge=0, le=1)] | Spectrum = 1
    full_well: int = 18_000
    serial_reg_full_well: int | None = None
    dark_current: PositiveFloat = Field(0.001, description="e/pix/sec")
    clock_induced_charge: PositiveFloat = Field(0, description="e/pix/sec")
    bit_depth: PositiveInt = 12
    offset: int = 100
    gain: PositiveFloat = 1
    name: str = ""

    # npixels_h: int = 1000
    # npixels_v: int = 1000

    # TODO: add photodiode size ... this needs to be reconciled with the up/down-scaling
    # that we do elsewhere in the simulation
    # photodiode_size: float = 6.5

    def apply_em_gain(self, electron_image: npt.NDArray) -> npt.NDArray:
        # default implementation does nothing
        return electron_image

    def apply_pre_quantization_binning(
        self, total_electrons: npt.NDArray, binning: int, mode: str = "sum"
    ) -> npt.NDArray:
        # default implementation does nothing, implemented in CCD types
        return total_electrons

    def quantize_electrons(
        self, total_electrons: npt.NDArray, xp: NumpyAPI
    ) -> npt.NDArray:
        voltage = xp.norm_rvs(total_electrons, self.read_noise) * self.gain
        return xp.maximum(xp.round((voltage / self.adc_gain) + self.offset), 0)  # type: ignore

    def apply_post_quantization_binning(
        self, gray_values: npt.NDArray, binning: int, mode: str = "sum"
    ) -> npt.NDArray:
        # default implementation does nothing, implemented in CMOS types
        return gray_values

    def simulate(
        self,
        photons_per_second: "xrDataArray",
        exposure_ms: "float | xrDataArray" = 100,
        binning: int = 1,
        add_poisson: bool = True,
        xp: "NumpyAPI | None" = None,
    ) -> "ArrayProtocol":
        xp = NumpyAPI.create(xp)

        exposure_s = exposure_ms / 1000
        incident_photons = photons_per_second * exposure_s
        # restrict to positive values
        incident_photons = xp.maximum(incident_photons.data, 0)

        # sample poisson noise
        if add_poisson:
            # FIXME: commenting this out since we also apply it in filtered_emission...
            # need to reconcile this
            # incident_photons = incident_photons * camera.qe
            detected_photons = xp.poisson_rvs(
                incident_photons, shape=incident_photons.shape
            )

        # dark current
        avg_dark_e = self.dark_current * exposure_s + self.clock_induced_charge
        if not isinstance(avg_dark_e, float):
            new_shape = avg_dark_e.shape + (1,) * (detected_photons.ndim - 1)
            avg_dark_e = np.asarray(avg_dark_e).reshape(new_shape)  # type: ignore [assignment]
        thermal_electrons = xp.poisson_rvs(avg_dark_e, shape=detected_photons.shape)
        total_electrons = detected_photons + thermal_electrons

        # cap total electrons to full-well-capacity
        total_electrons = xp.minimum(total_electrons, self.full_well)

        if binning > 1:
            total_electrons = self.apply_pre_quantization_binning(
                total_electrons, binning
            )

        # add em gain
        total_electrons = self.apply_em_gain(total_electrons)

        # cap total electrons to serial register full-well-capacity
        if self.serial_reg_full_well is not None:
            total_electrons = xp.minimum(total_electrons, self.serial_reg_full_well)

        # model read noise
        gray_values = self.quantize_electrons(total_electrons, xp)

        # sCMOS binning
        if binning > 1:
            gray_values = self.apply_post_quantization_binning(gray_values, binning)

        # ADC saturation
        gray_values = xp.minimum(gray_values, self.max_intensity)
        if self.bit_depth > 16:
            output = gray_values.astype("uint32")
        if self.bit_depth > 8:
            output = gray_values.astype("uint16")
        else:
            output = gray_values.astype("uint8")

        return output  # type: ignore[no-any-return]

    @property
    def dynamic_range(self) -> float:
        return self.full_well / self.read_noise

    def render(
        self,
        image: xrDataArray,
        exposure_ms: float | xrDataArray = 100,
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
        new_data = self.simulate(
            photons_per_second=image,
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


class CameraCCD(_Camera):
    camera_type: Literal["CCD"] = "CCD"

    def apply_pre_quantization_binning(
        self, total_electrons: npt.NDArray, binning: int, mode: str = "sum"
    ) -> npt.NDArray:
        return bin_window(total_electrons, binning, mode)


class CameraEMCCD(_Camera):
    camera_type: Literal["EMCCD"] = "EMCCD"

    em_gain: float

    def apply_pre_quantization_binning(
        self, total_electrons: npt.NDArray, binning: int, mode: str = "sum"
    ) -> npt.NDArray:
        return bin_window(total_electrons, binning, mode)

    def apply_em_gain(self, electron_image: npt.NDArray) -> npt.NDArray:
        from scipy import stats

        # FIXME: is there a more elegant way to deal with gamma rvs with shape = 0?
        ind_zero = electron_image <= 0
        electron_image[ind_zero] += 1
        # gamma shape is input electrons / scale is EM gain
        electron_image = stats.gamma.rvs(
            electron_image.astype(float), scale=self.em_gain
        )
        electron_image[ind_zero] = 0
        electron_image = np.round(electron_image).astype(int)

        return electron_image


class CameraCMOS(_Camera):
    camera_type: Literal["CMOS"] = "CMOS"

    def apply_post_quantization_binning(
        self, gray_values: npt.NDArray, binning: int, mode: str = "mean"
    ) -> npt.NDArray:
        return bin_window(gray_values, binning, mode)
