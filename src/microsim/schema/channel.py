from typing import Any

from pydantic import BaseModel, field_validator


class WaveBand(BaseModel):
    wavelength: float
    bandwidth: float


def _v_waveband(cls, value: Any):
    if isinstance(value, float | int):
        value = {"wavelength": value, "bandwidth": 1}
    return value


class Channel(BaseModel):
    name: str
    excitation: WaveBand
    emission: WaveBand

    _v_ex = field_validator("excitation", mode="before")(_v_waveband)
    _v_em = field_validator("emission", mode="before")(_v_waveband)
