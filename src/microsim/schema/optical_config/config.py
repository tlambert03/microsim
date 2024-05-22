from typing import Any

from pydantic import Field, model_validator

from microsim.schema._base_model import SimBaseModel
from microsim.schema.spectrum import Spectrum

from .filter import Bandpass, Filter, FilterPlacement


def _validate_filter(cls: type, value: Any) -> Any:
    if isinstance(value, float | int):
        value = {"bandcenter": value, "bandwidth": 1}
    return value


class OpticalConfig(SimBaseModel):
    name: str = ""
    filters: list[FilterPlacement] = Field(default_factory=list)

    @property
    def excitation(self) -> Filter | None:
        return next(f for f in self.filters if f.path == "EX")

    @property
    def emission(self) -> Filter | None:
        return next(f for f in self.filters if f.path == "EM")

    # excitation: Filter
    # emission: Filter
    # beam_splitter: Filter | None = None

    # cast integers to bandpass filters with bandwidth=1
    # TODO: could move to a base class
    # _v_ex = field_validator("excitation", mode="before")(_validate_filter)
    # _v_em = field_validator("emission", mode="before")(_validate_filter)
    # _v_bs = field_validator("beam_splitter", mode="before")(_validate_filter)

    @classmethod
    def from_fpbase(
        cls, microscope_id: str, config_name: str | None = None
    ) -> "OpticalConfig":
        from microsim.fpbase import get_microscope

        if config_name is None:
            if "::" not in microscope_id:
                raise ValueError(
                    "If config_name is not provided, microscope_id must be "
                    "in the form 'scope::config'"
                )
            microscope_id, config_name = microscope_id.split("::")

        fpbase_scope = get_microscope(microscope_id)
        for cfg in fpbase_scope.opticalConfigs:
            if cfg.name.lower() == config_name.lower():
                return cls(
                    name=cfg.name,
                    filters=[
                        FilterPlacement(
                            path=f.path,
                            reflects=f.reflects,
                            spectrum=Spectrum.from_fpbase(f.spectrum),
                            name=f.name,
                            type=f.spectrum.subtype,
                        )
                        for f in cfg.filters
                    ],
                )

        raise ValueError(
            f"Could not find config named {config_name!r} in FPbase microscope "
            f"{microscope_id!r}. Available names: "
            f"{', '.join(repr(c.name) for c in fpbase_scope.opticalConfigs)}"
        )

    @model_validator(mode="before")
    def _vmodel(cls, value: Any) -> Any:
        if isinstance(value, str):
            if "::" not in value:
                raise ValueError(
                    "If OpticalConfig is provided as a string, it must be "
                    "in the form 'fpbase_scope_id::config_name'"
                )
            # TODO: seems weird to have to cast back to dict...
            # but otherwise doesn't work with 'before' validator.  look into it.
            return cls.from_fpbase(value).model_dump()
        return value


# class FITC(OpticalConfig):
#     name: str = "FITC"
#     filters: list[FilterPlacement] = []
#     excitation: Filter = Bandpass(bandcenter=488, bandwidth=1)
#     emission: Filter = Bandpass(bandcenter=525, bandwidth=50)


FITC = OpticalConfig(
    name="FITC",
    filters=[
        FilterPlacement(path="EX", spectrum=Spectrum(wavelength=[488], intensity=[1])),
        FilterPlacement(path="EM", spectrum=Spectrum(wavelength=[525], intensity=[1])),
    ],
)
