import random
from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from microsim._field_types import FloatDtype

from ._base_model import SimBaseModel
from .backend import BackendName, DeviceName, NumpyAPI


class InMemoryCacheSizes(SimBaseModel):
    """Parameters that control the in-memory cache size of various cached functions."""

    psf: int = Field(
        default=64,
        description=(
            "The maximum number of PSFs that will be stored in the in-memory cache, "
            "which follows the LRU caching strategy. Note, this setting will only take "
            "effect by modifying the equivalent environment variable prior to "
            "importing microsim."
        ),
        frozen=True,
    )
    otf_mb: float = Field(
        default=1000,
        description=(
            "The maximum total size (in MB) of Fourier-transformed PSFs that will be "
            "kept in memory (LRU) for reuse across simulations with the same optics. "
            "Set to 0 to disable. Unaffected by `cache.read`/`cache.write`, which "
            "control on-disk caching. For GPU backends, this is GPU memory."
        ),
    )


class CacheSettings(SimBaseModel):
    """On-disk cache settings for ground truth.

    A boolean may be passed instead, to set both `read` and `write`.

    Parameters
    ----------
    read : bool
        Read cached ground truth, if available.
    write : bool
        Write generated ground truth to the cache.
    in_mem_size : InMemoryCacheSizes
        Sizes of the in-memory caches.
    """

    read: bool = True
    write: bool = True
    in_mem_size: InMemoryCacheSizes = Field(default_factory=InMemoryCacheSizes)

    @model_validator(mode="before")
    @classmethod
    def _vmodel(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            value = value.lower() not in {"false", "0", "no", "off", "n", "none"}
        return {"read": bool(value), "write": bool(value)}


class Settings(SimBaseModel, BaseSettings):
    """Settings that affect how a simulation is computed.

    Every setting can also be set with an environment variable `MICROSIM_<NAME>`
    (nested settings separated by `__`, e.g. `MICROSIM_CACHE__WRITE=0`).

    Parameters
    ----------
    np_backend : str
        Array library: `"numpy"`, `"cupy"`, `"jax"`, `"torch"`, or `"auto"` (CuPy
        if it can be imported, otherwise NumPy).
    device : str
        Not used yet.
    random_seed : int, optional
        Seed for random ground truth and noise.  A random seed is chosen if not
        given.
    cache : CacheSettings | bool
        Whether to read and write the on-disk ground truth cache.
    """

    np_backend: BackendName = "auto"
    device: DeviceName = "auto"
    float_dtype: FloatDtype = Field(
        "float32",  # type: ignore[arg-type]
        description="Floating-point precision to use for simulations.",
    )
    random_seed: int | None = Field(
        default_factory=lambda: random.randint(0, 2**32 - 1)
    )
    max_psf_radius_aus: float | Literal["auto"] | None = Field(
        "auto",
        description=(
            "Restrict the lateral radius of simulated PSFs to this many Airy units. "
            "`'auto'` (default) is our opinionated balance between speed and "
            "accuracy, aiming for ~1% RMSE (and <2% max error) relative to an "
            "untruncated PSF, which is below shot noise unless pixels collect well "
            "over ~10^4 photons. For widefield, the radius grows with the depth of the "
            "simulated volume, since out-of-focus light spreads in a cone of "
            "half-angle asin(NA/n): it is ~14 AU (for the tails of the in-focus Airy "
            "pattern) plus half the radius of that cone at half the volume depth "
            "(e.g. ~17 AU for a 1 um deep volume, ~27 AU for 5 um, at NA 1.4). For "
            "confocal, where the pinhole rejects out-of-focus light, it is 8 AU. A "
            "number sets the radius directly; a too-small radius loses out-of-focus "
            "light (in a 5 um deep widefield volume, 8 AU loses ~20% of it). `None` "
            "uses the full extent of the simulated space (most accurate, but much "
            "slower for large volumes: ~2-4x slower per sample than 'auto' at "
            "2048x2048 pixels). The PSF is never larger than the simulated space."
        ),
    )
    cache: CacheSettings = Field(default_factory=CacheSettings)
    spectral_bins_per_emission_channel: int = Field(
        1,
        description="Number of wavelengths to use (per channel) when simulating the "
        "optical image. By default, a single centroid wavelength is used to "
        "approximate the emission wavelength.  Increasing this will make the "
        "simulation more realistic, by superposing multiple different PSFs, but will "
        "also increase memory usage and time.",
    )
    excitation_bin_width_nm: float = Field(
        10,
        gt=0,
        description="Maximum width (nm) of the wavelength bins used to represent the "
        "excitation light absorbed by each fluorophore, for modalities that focus the "
        "excitation (e.g. confocal).  Each bin uses a separate excitation spot, so "
        "laser lines further apart than this are always simulated separately.",
    )
    spectral_bin_threshold_percentage: float = Field(
        1,
        description="Percentage of the peak intensity to use as a threshold when "
        "binning the emission (and excitation) spectra. This is used to determine the "
        "range of wavelengths to consider when simulating the optical image.  First, "
        "the spectrum is trimmed to include only relevant wavelengths (those with "
        "intensity above this threshold). Then, the spectrum is discretized into "
        "spectral_bins_per_emission_channel bins (or, for excitation, bins of at most "
        "excitation_bin_width_nm).",
    )

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        validate_assignment=True,
        # this allows any of these to be set via environment variables: MICROSIM_<name>
        env_prefix="MICROSIM_",
        env_nested_delimiter="__",
    )

    def backend_module(self) -> NumpyAPI:
        backend = NumpyAPI.create(self.np_backend)
        backend.float_dtype = self.float_dtype
        if self.random_seed is not None:
            backend.set_random_seed(self.random_seed)
        return backend
