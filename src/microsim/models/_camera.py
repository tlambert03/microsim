import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field
from scipy import stats


class Camera(BaseModel):
    photodiode_size: float
    qe: float  # TODO: spectrum
    full_well: int
    # TODO: serial register fullwell?
    dark_current: float = Field(..., description="e/pix/sec")
    clock_induced_charge: float
    read_noise: float  # as function of readout rate?
    bit_depth: int
    offset: int
    gain: float = 1
    readout_rate: float = Field(..., description="MHz")
    npixels_h: int = 1000
    npixels_v: int = 1000

    # binning?  or keep in simulate

    @property
    def dynamic_range(self) -> float:
        return self.full_well / self.read_noise

    def apply_em_gain(self, electron_image: npt.NDArray) -> npt.NDArray:
        return electron_image

    def simulate(
        self,
        image: np.ndarray,
        exposure: float = 100,
        binning: int = 1,
        add_poisson: bool = True,
    ) -> np.ndarray:
        """Simulate imaging process.

        Parameters
        ----------
        image : np.ndarray
            array where each element represents photons / second
        exposure : float, optional
            _description_, by default 100
        binning : int, optional
            _description_, by default 1
        add_poisson : bool, optional
            _description_, by default True
        """
        from microsim.simulate import simulate_camera

        return simulate_camera(
            self, image, exposure, binning=binning, add_poisson=add_poisson
        )

    @property
    def adc_gain(self) -> float:
        # assume ADC has been set such that voltage at FWC matches max bit depth
        return self.full_well / self.max_intensity

    @property
    def max_intensity(self) -> int:
        return int(2**self.bit_depth - 1)

    def quantize_electrons(self, total_electrons: npt.NDArray) -> npt.NDArray:
        voltage = stats.norm.rvs(total_electrons, self.read_noise) * self.gain
        return np.round((voltage / self.adc_gain) + self.offset)  # type: ignore


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
    photodiode_size=6.45,
    qe=0.70,
    gain=1,
    full_well=18000,
    dark_current=0.0005,
    clock_induced_charge=1,
    read_noise=6,
    readout_rate=14,
    bit_depth=12,
    offset=100,
    npixels_h=1344,
    npixels_v=1024,
)
