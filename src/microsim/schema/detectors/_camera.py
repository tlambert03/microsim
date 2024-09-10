from typing import Annotated

import numpy as np
import numpy.typing as npt
from annotated_types import Ge, Interval
from pydantic import Field
from scipy import stats

from microsim._data_array import DataArray, xrDataArray
from microsim.schema._base_model import SimBaseModel
from microsim.schema.backend import NumpyAPI
from microsim.schema.spectrum import Spectrum

PositiveFloat = Annotated[float, Ge(0)]
PositiveInt = Annotated[int, Ge(0)]


class Camera(SimBaseModel):
    read_noise: PositiveFloat = 6  # as function of readout rate?
    qe: Annotated[float, Interval(ge=0, le=1)] | Spectrum = 1  # TODO: spectrum
    full_well: int = 18_000
    # TODO: serial register fullwell?
    dark_current: PositiveFloat = Field(0.001, description="e/pix/sec")
    clock_induced_charge: PositiveFloat = Field(0, description="e/pix/sec")
    bit_depth: PositiveInt = 12
    offset: int = 100
    gain: PositiveFloat = 1
    readout_rate: PositiveFloat = Field(1, description="MHz")
    name: str = ""
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


# fmt: off
r2qe = [0.5188,0.5219,0.5249,0.5279,0.5307,0.5335,0.5363,0.539,0.5418,0.5445,0.5473,0.5501,0.5531,0.5561,0.5593,0.5627,0.5662,0.57,0.5738,0.5778,0.5819,0.586,0.5902,0.5943,0.5984,0.6024,0.6064,0.6102,0.6138,0.6172,0.6204,0.6234,0.6261,0.6286,0.631,0.6333,0.6356,0.6378,0.6401,0.6425,0.6451,0.6478,0.6508,0.654,0.6573,0.6607,0.6641,0.6675,0.6708,0.6739,0.6769,0.6796,0.682,0.6841,0.6859,0.6876,0.6891,0.6905,0.6919,0.6933,0.6948,0.6964,0.698,0.6997,0.7013,0.7027,0.7041,0.7053,0.7062,0.7068,0.7072,0.7072,0.707,0.7065,0.7058,0.7049,0.7039,0.7027,0.7014,0.7001,0.6988,0.6976,0.6964,0.6954,0.6946,0.6939,0.6934,0.6931,0.693,0.693,0.6931,0.6932,0.6932,0.6931,0.6928,0.6923,0.6914,0.6902,0.6886,0.6864,0.6837,0.6805,0.6769,0.673,0.6689,0.6648,0.6608,0.657,0.6536,0.6506,0.6481,0.6459,0.6442,0.6426,0.6413,0.6401,0.639,0.6378,0.6365,0.6351,0.6334,0.6315,0.6294,0.6272,0.6249,0.6224,0.6199,0.6174,0.6148,0.6123,0.6099,0.6075,0.6052,0.603,0.6009,0.599,0.5971,0.5954,0.5938,0.5923,0.5908,0.5894,0.588,0.5866,0.5853,0.5839,0.5826,0.5811,0.5797,0.5781,0.5765,0.5748,0.573,0.571,0.5689,0.5667,0.5645,0.5624,0.5604,0.5586,0.557,0.5556,0.5543,0.553,0.5517,0.5505,0.5491,0.5476,0.546,0.5442,0.5422,0.5401,0.5379,0.5356,0.5335,0.5314,0.5294,0.5275,0.5257,0.5238,0.522,0.5202,0.5184,0.5165,0.5146,0.5127,0.5107,0.5086,0.5065,0.5043,0.5021,0.4998,0.4974,0.4949,0.4923,0.4897,0.487,0.4842,0.4813,0.4784,0.4754,0.4724,0.4694,0.4664,0.4634,0.4605,0.4576,0.4548,0.4521,0.4494,0.4469,0.4445,0.4421,0.4399,0.4377,0.4356,0.4335,0.4315,0.4296,0.4277,0.4259,0.424,0.4223,0.4205,0.4188,0.4171,0.4154,0.4137,0.4119,0.4101,0.4083,0.4065,0.4045,0.4025,0.4005,0.3983,0.3961,0.3938,0.3914,0.389,0.3865,0.3841,0.3816,0.3792,0.3767,0.3743,0.372,0.3697,0.3674,0.3653,0.3633,0.3614,0.3595,0.3578,0.3561,0.3545,0.3529,0.3513,0.3497,0.3481,0.3464,0.3446,0.3427,0.3408,0.3387,0.3366,0.3345,0.3324,0.3303,0.3282,0.3262,0.3243,0.3225,0.3208,0.3193,0.3178,0.3164,0.3149,0.3134,0.3118,0.3101,0.3082,0.3062,0.3041,0.3019,0.2996,0.2974,0.2951,0.293,0.2909,0.289,0.2873,0.2856,0.2842,0.2828,0.2815,0.2803,0.2792,0.2781,0.277,0.2759,0.2748,0.2737,0.2725,0.2712,0.2699,0.2684,0.267,0.2654,0.2638,0.2621,0.2604,0.2586,0.2569,0.255,0.2532,0.2514,0.2495,0.2477,0.2459,0.2441,0.2423,0.2405,0.2388,0.2371,0.2355,0.2339,0.2324,0.2309,0.2295,0.2281,0.2267,0.2254,0.2241,0.2228,0.2215,0.2202,0.2189,0.2176,0.2162,0.2149,0.2135,0.2121,0.2106,0.2092,0.2076,0.2061,0.2045,0.2029,0.2012,0.1996,0.198,0.1964,0.1948,0.1933,0.1918,0.1903,0.1889,0.1875,0.1862,0.185,0.1839,0.1828,0.1819,0.181,0.1803,0.1796,0.1789,0.1782,0.1775,0.1768,0.176,0.1752,0.1742,0.1731,0.1719,0.1704,0.1689,0.1672,0.1654,0.1635,0.1616,0.1596,0.1577,0.1558,0.154,0.1523,0.1507,0.1492,0.1478,0.1464,0.1452,0.144,0.1429,0.1418,0.1408,0.1399,0.139,0.1381,0.1372,0.1364,0.1355,0.1347,0.1339,0.133,0.1321,0.1312,0.1303,0.1293,0.1283,0.1273,0.1262,0.1252,0.1241,0.123,0.1218,0.1207,0.1195,0.1184,0.1172,0.1161,0.1149,0.1138,0.1126,0.1115,0.1104,0.1093,0.1082,0.1071,0.106,0.105,0.104,0.1029,0.1019,0.1009,0.0999,0.099,0.098,0.0971,0.0962,0.0952,0.0943,0.0935,0.0926,0.0917,0.0909,0.09,0.0892,0.0884,0.0876,0.0868,0.086,0.0852,0.0844,0.0837,0.0829,0.0822,0.0815,0.0807,0.08,0.0793,0.0786,0.0779,0.0772,0.0766,0.0759,0.0752,0.0746,0.0739,0.0732,0.0726,0.0719,0.0713,0.0707,0.07,0.0694,0.0688,0.0681,0.0675,0.0669,0.0663,0.0656,0.065,0.0644,0.0638,0.0632,0.0626,0.0619,0.0613,0.0607,0.0601,0.0595,0.0589,0.0583,0.0577,0.057,0.0564,0.0558,0.0552,0.0546,0.054,0.0534,0.0527,0.0521,0.0515,0.0509,0.0503,0.0496,0.049,0.0484,0.0477,0.0471,0.0465,0.0458,0.0452,0.0445,0.0439,0.0432,0.0426,0.0419,0.0413,0.0406,0.04,0.0393,0.0387,0.038,0.0374,0.0367,0.0361,0.0355,0.0349,0.0343,0.0337,0.0331,0.0325,0.0319,0.0314,0.0308,0.0303,0.0298,0.0293,0.0288,0.0283,0.0278,0.0274,0.027,0.0266,0.0262,0.0258,0.0254,0.025,0.0247,0.0243,0.024,0.0237,0.0233,0.023,0.0227,0.0224,0.0221,0.0218,0.0215,0.0213,0.021,0.0207,0.0204,0.0201,0.0198,0.0195,0.0193,0.019,0.0187,0.0184,0.0181,0.0178,0.0174,0.0171,0.0168,0.0165,0.0162,0.0159,0.0155,0.0152,0.0149,0.0146,0.0142,0.0139,0.0136,0.0133,0.013,0.0127,0.0124,0.0121,0.0118,0.0115,0.0112,0.0109,0.0106,0.0103,0.0101,0.0098]  # noqa
# fmt: on
OrcaR2QE = Spectrum(wavelength=np.arange(400, 400 + len(r2qe)), intensity=r2qe)

ICX285 = CameraCCD(
    # photodiode_size=6.45,
    name="ICX285",
    qe=OrcaR2QE,
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
