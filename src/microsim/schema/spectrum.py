from collections.abc import Iterable
from typing import Any

import numpy as np
import pint
from pydantic import field_validator, model_validator

from microsim._field_types import Nanometers, NumpyNdarray

from ._base_model import SimBaseModel


class _AryRepr:
    def __init__(self, obj: np.ndarray | pint.Quantity) -> None:
        self.dtype = obj.dtype
        self.shape = obj.shape
        self.units = getattr(obj, "units", None)

    def __repr__(self) -> str:
        unit = f" units={self.units}" if self.units else ""
        return f"ndarray<shape={self.shape} dtype={self.dtype}{unit}>"


class Spectrum(SimBaseModel):
    wavelength: Nanometers
    intensity: NumpyNdarray  # normalized to 1
    scalar: float = 1  # scalar to multiply intensity by, such as EC or QY

    def __repr_args__(self) -> Iterable[tuple[str | None, Any]]:
        for _fname, _val in super().__repr_args__():
            if isinstance(_val, pint.Quantity | np.ndarray):
                _val = _AryRepr(_val)
            yield _fname, _val

    @property
    def peak_wavelength(self) -> Nanometers:
        """Wavelength corresponding to maximum intensity."""
        return self.wavelength[np.argmax(self.intensity)]  # type: ignore

    @field_validator("intensity", mode="after")
    @classmethod
    def _validate_intensity(cls, value: np.ndarray) -> np.ndarray:
        if not np.all(value >= 0):
            raise ValueError("Intensity must be non-negative")
        if not 0.9 <= np.max(value) <= 1:
            raise ValueError("Intensity must be normalized to 1")
        return value

    @model_validator(mode="before")
    def _cast_spectrum(cls, value: Any) -> Any:
        if isinstance(value, list | tuple):
            data = np.asarray(value)
            if not data.ndim == 2:
                raise ValueError("Spectrum data must be 2D")
            if not data.shape[1] == 2:
                raise ValueError("Spectrum data must have two columns")
            return {"wavelength": data[:, 0], "intensity": data[:, 1]}
        if isinstance(value, dict):
            if "wavelength" in value and "intensity" in value:
                if not len(value["wavelength"]) == len(value["intensity"]):
                    raise ValueError(
                        "Wavelength and intensity must have the same length"
                    )
        return value
