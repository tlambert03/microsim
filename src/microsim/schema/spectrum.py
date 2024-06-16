import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import pint
from pydantic import field_validator, model_validator

from microsim._field_types import Nanometers, NumpyNdarray

from ._base_model import SimBaseModel

if TYPE_CHECKING:
    from microsim.fpbase import Spectrum as FPbaseSpectrum


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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Spectrum):
            return False
        return (
            np.allclose(self.intensity, other.intensity)
            and np.allclose(self.wavelength.magnitude, other.wavelength.magnitude)
            and self.scalar == other.scalar
        )

    def __array__(self) -> np.ndarray:
        return np.column_stack((self.wavelength.magnitude, self.intensity))

    def __index__(self) -> int:
        return id(self)

    def inverted(self) -> "Spectrum":
        return self.model_copy(update={"intensity": 1 - self.intensity})

    def integral(self) -> float | pint.Quantity:
        return np.trapz(self.intensity, self.wavelength.magnitude)  # type: ignore [no-any-return]

    def __mul__(self, other: "float | pint.Quantity | Spectrum") -> "Spectrum":
        if isinstance(other, Spectrum):
            return self._intensity_op(other, np.multiply)
        return self.model_copy(update={"intensity": self.intensity * other})

    def __truediv__(self, other: "float | pint.Quantity | Spectrum") -> "Spectrum":
        if isinstance(other, Spectrum):
            return self._intensity_op(other, np.true_divide)
        return self.model_copy(update={"intensity": self.intensity / other})

    def _intensity_op(self, other: "Spectrum", op: np.ufunc) -> "Spectrum":
        slc1, slc2 = get_overlapping_indices(
            self.wavelength.magnitude, other.wavelength.magnitude
        )
        intens1 = self.intensity[slc1]
        intens2 = other.intensity[slc2]
        return self.model_copy(
            update={
                "intensity": op(intens1, intens2),
                "wavelength": self.wavelength[slc1],  # type: ignore[index]
            }
        )

    @classmethod
    def from_fpbase(cls, spectrum: "FPbaseSpectrum") -> "Spectrum":
        data = np.asarray(spectrum.data)
        return cls(wavelength=data[:, 0], intensity=data[:, 1])

    def __repr_args__(self) -> Iterable[tuple[str | None, Any]]:
        for _fname, _val in super().__repr_args__():
            if isinstance(_val, pint.Quantity | np.ndarray):
                _val = _AryRepr(_val)
            yield _fname, _val

    @property
    def peak_wavelength(self) -> Nanometers:
        """Wavelength corresponding to maximum intensity."""
        return self.wavelength[np.argmax(self.intensity)]  # type: ignore

    @property
    def max_intensity(self) -> pint.Quantity | float:
        """Maximum intensity."""
        return np.max(self.intensity)  # type: ignore

    @field_validator("intensity", mode="after")
    @classmethod
    def _validate_intensity(cls, value: np.ndarray) -> np.ndarray:
        if not np.all(value >= 0):
            warnings.warn(
                "Clipping negative intensity values in spectrum to 0", stacklevel=2
            )
            value = np.clip(value, 0, None)
        if not 0.9 <= np.max(value) <= 1:
            warnings.warn("Normalize intensity to 1", stacklevel=2)
            value = value / np.max(value)
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

    def plot(self, show: bool = True) -> None:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(111)

        ax.plot(self.wavelength.magnitude, self.intensity)
        if show:
            plt.show()


def get_overlapping_indices(ary1: np.ndarray, ary2: np.ndarray) -> tuple[slice, slice]:
    """Return slices of overlapping subset of arrays.

    This assumes that the arrays are sorted 1d arrays.
    """
    # Find the indices of the start and end of the overlapping subset
    start = max(ary1[0], ary2[0])
    end = min(ary1[-1], ary2[-1])

    # Find the indices of the start and end of the overlapping subset
    start_idx = np.searchsorted(ary1, start)
    end_idx = np.searchsorted(ary1, end, side="right")

    start_idx2 = np.searchsorted(ary2, start)
    end_idx2 = np.searchsorted(ary2, end, side="right")
    return slice(start_idx, end_idx), slice(start_idx2, end_idx2)
