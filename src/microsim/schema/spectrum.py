from collections.abc import Iterable
from typing import Any

import numpy as np
from pydantic import model_validator

from ._base_model import SimBaseModel
from ._types import NumpyNdarray


class _AryRepr:
    def __init__(self, dtype: Any, shape: Any) -> None:
        self.dtype = dtype
        self.shape = shape

    def __repr__(self) -> str:
        return f"ndarray<shape={self.shape} dtype={self.dtype}>"


class Spectrum(SimBaseModel):
    wavelength: NumpyNdarray  # nm
    intensity: NumpyNdarray  # normalized to 1
    scalar: float = 1  # scalar to multiply intensity by, such as EC or QY

    def __repr_args__(self) -> Iterable[tuple[str | None, Any]]:
        for _fname, _val in super().__repr_args__():
            if isinstance(_val, np.ndarray):
                _val = _AryRepr(_val.dtype, _val.shape)
            yield _fname, _val

    @property
    def peak_wavelength(self) -> float:
        """Wavelength corresponding to maximum intensity."""
        return float(self.wavelength[np.argmax(self.intensity)])

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
