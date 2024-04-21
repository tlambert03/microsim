from typing import Any

from pydantic import field_validator

from microsim.schema._base_model import SimBaseModel

from .filter import Bandpass, Filter


def _validate_filter(cls: type, value: Any) -> Any:
    if isinstance(value, float | int):
        value = {"bandcenter": value, "bandwidth": 1}
    return value


class OpticalConfig(SimBaseModel):
    name: str
    excitation: Filter
    emission: Filter
    beam_splitter: Filter | None = None

    # cast integers to bandpass filters with bandwidth=1
    # TODO: could move to a base class
    _v_ex = field_validator("excitation", mode="before")(_validate_filter)
    _v_em = field_validator("emission", mode="before")(_validate_filter)
    _v_bs = field_validator("beam_splitter", mode="before")(_validate_filter)


class FITC(OpticalConfig):
    name: str = "FITC"
    excitation: Filter = Bandpass(bandcenter=488, bandwidth=1)
    emission: Filter = Bandpass(bandcenter=525, bandwidth=50)
