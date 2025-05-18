import random
from typing import Any, ClassVar

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


class CacheSettings(SimBaseModel):
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
    np_backend: BackendName = "auto"
    device: DeviceName = "auto"
    float_dtype: FloatDtype = Field(
        "float32",  # type: ignore[arg-type]
        description="Floating-point precision to use for simulations.",
    )
    random_seed: int | None = Field(
        default_factory=lambda: random.randint(0, 2**32 - 1)
    )
    max_psf_radius_aus: float | None = Field(
        6,
        description=(
            "When simulating, restrict generated lateral PSF size to no more than this "
            "many Airy units. Decreasing this can *dramatically* speed up simulations, "
            "but will decrease accuracy. If `None`, no restriction is applied, and the "
            "psf will be generated to the full extent of the simulated space."
        ),
    )
    cache: CacheSettings = Field(default_factory=CacheSettings)
    spectral_bins_per_emission_channel: int = Field(
        1,
        description="Number of wavelengths to use (per channel) when simulating the "
        "optical image. By default, a single centroid wavelength is used to approximate"
        "the emission wavelength.  Increasing this will make the simulation more "
        "realistic, by superposing multiple different PSFs, but will also increase "
        "memory usage and time.",
    )
    spectral_bin_threshold_percentage: float = Field(
        1,
        description="Percentage of the total intensity to use as a threshold when "
        "binning the emission spectrum. This is used to determine the range of "
        "wavelengths to consider when simulating the optical image.  First, the "
        "spectrum is trimmed to include only relevant wavelengths (those with "
        "intensity above this threshold). Then, the spectrum is discretized into "
        "spectral_bins_per_emission_channel bins.",
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
