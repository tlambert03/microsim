import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import pint
import xarray as xr
from pydantic import field_validator, model_validator

from microsim._field_types import NumpyNdarray

from ._base_model import SimBaseModel

if TYPE_CHECKING:
    from xarray.plot.accessor import DataArrayPlotAccessor

    from microsim.fpbase import Spectrum as FPbaseSpectrum


class _AryRepr:
    def __init__(self, obj: np.ndarray) -> None:
        self.dtype = obj.dtype
        self.shape = obj.shape

    def __repr__(self) -> str:
        return f"ndarray<shape={self.shape} dtype={self.dtype}>"


class Spectrum(SimBaseModel):
    wavelength: NumpyNdarray  # nanometers
    intensity: NumpyNdarray  # normalized to 1
    scalar: float = 1  # scalar to multiply intensity by, such as EC or QY

    @field_validator("intensity", mode="after")
    @classmethod
    def _validate_intensity(cls, value: np.ndarray) -> np.ndarray:
        if not np.all(value >= 0):
            warnings.warn(
                "Clipping negative intensity values in spectrum to 0", stacklevel=2
            )
            value = np.clip(value, 0, None)
        # if not 0.9 <= np.max(value) <= 1:
        #     warnings.warn("Normalize intensity to 1", stacklevel=2)
        #     value = value / np.max(value)
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
                    breakpoint()
                    raise ValueError(
                        "Wavelength and intensity must have the same length"
                    )
        return value

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Spectrum):
            return False
        return (
            np.allclose(self.intensity, other.intensity)
            and np.allclose(self.wavelength, other.wavelength)
            and self.scalar == other.scalar
        )

    def __array__(self) -> np.ndarray:
        return np.column_stack((self.wavelength, self.intensity))

    def as_xarray(self) -> "xr.DataArray":
        from .dimensions import Axis

        return xr.DataArray(
            self.intensity,
            coords={
                Axis.W: xr.DataArray(
                    self.wavelength,
                    dims=[Axis.W],
                    attrs={"units": "nm"},
                )
            },
            dims=[Axis.W],
            name="intensity",
        )

    def __index__(self) -> int:
        return id(self)

    def inverted(self) -> "Spectrum":
        return self.model_copy(update={"intensity": 1 - self.intensity})

    def integral(self) -> float:
        return np.trapz(self.intensity, self.wavelength)  # type: ignore [no-any-return]

    def __mul__(self, other: "float | pint.Quantity | Spectrum") -> "Spectrum":
        if isinstance(other, pint.Quantity):
            other = other.magnitude
        if isinstance(other, Spectrum):
            return self._intensity_op(other, np.multiply)
        return self.model_copy(update={"intensity": self.intensity * other})

    def __add__(self, other: "float | pint.Quantity | Spectrum") -> "Spectrum":
        if isinstance(other, pint.Quantity):
            other = other.magnitude
        if isinstance(other, Spectrum):
            return self._intensity_op(other, np.add)
        return self.model_copy(update={"intensity": self.intensity + other})

    def __truediv__(self, other: "float | pint.Quantity | Spectrum") -> "Spectrum":
        if isinstance(other, pint.Quantity):
            other = other.magnitude
        if isinstance(other, Spectrum):
            return self._intensity_op(other, np.true_divide)
        return self.model_copy(update={"intensity": self.intensity / other})

    def _intensity_op(self, other: "Spectrum", op: np.ufunc) -> "Spectrum":
        selfx, otherx = xr.align(self.as_xarray(), other.as_xarray(), join="outer")
        selfx = selfx.fillna(0)
        otherx = otherx.fillna(0)
        new = op(selfx, otherx)
        return self.model_copy(
            update={
                "intensity": new,
                "wavelength": new.coords["w"].values,
            }
        )

    @classmethod
    def from_fpbase(cls, spectrum: "FPbaseSpectrum") -> "Spectrum":
        data = np.asarray(spectrum.data)
        return cls(wavelength=data[:, 0], intensity=data[:, 1])

    def __repr_args__(self) -> Iterable[tuple[str | None, Any]]:
        for _fname, _val in super().__repr_args__():
            if isinstance(_val, np.ndarray):
                _val = _AryRepr(_val)
            yield _fname, _val

    @property
    def peak_wavelength(self) -> float:
        """Wavelength corresponding to maximum intensity."""
        return float(self.wavelength[np.argmax(self.intensity)])

    @property
    def max_intensity(self) -> float:
        """Maximum intensity."""
        return np.max(self.intensity)  # type: ignore [no-any-return]

    @property
    def plot(self) -> "DataArrayPlotAccessor":
        return self.as_xarray().plot


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
