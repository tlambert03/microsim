from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeVar

import numpy as np
from pydantic import (
    GetCoreSchemaHandler,
    ValidatorFunctionWrapHandler,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_core import CoreSchema, core_schema

from microsim._data_array import ArrayProtocol, DataArray, xrDataArray

from ._base_model import SimBaseModel
from .dimensions import Axis


class FloatArray(Sequence[float]):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        list_schema = core_schema.list_schema(core_schema.float_schema())
        return core_schema.no_info_wrap_validator_function(
            cls._wrap_validator, list_schema
        )

    @classmethod
    def _wrap_validator(
        self, value: Any, handler: ValidatorFunctionWrapHandler
    ) -> np.ndarray:
        return np.array(value, dtype=np.float64)


class SpaceProtocol(Protocol):
    @property
    def axes(self) -> tuple[str, ...]: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def scale(self) -> tuple[float, ...]: ...


ArrayType = TypeVar("ArrayType")


class _Space(SimBaseModel):
    def rescale(self, img: xrDataArray) -> xrDataArray:
        return img

    def create(
        self: SpaceProtocol,
        array_creator: Callable[[Sequence[int]], ArrayProtocol] = np.zeros,
    ) -> xrDataArray:
        from microsim.util import uniformly_spaced_coords

        coords = uniformly_spaced_coords(self.shape, self.scale, axes=self.axes)
        data = array_creator(self.shape)
        attrs = {"space": self}
        return DataArray(data, coords=coords, dims=self.axes, name="space", attrs=attrs)

    @property
    def coords(self: SpaceProtocol) -> dict[str, FloatArray]:
        return {
            ax: np.arange(sh) * sc  # type: ignore
            for ax, sh, sc in zip(self.axes, self.shape, self.scale, strict=False)
        }


class _AxesSpace(_Space):
    axes: tuple[Axis, ...] = (Axis.Z, Axis.Y, Axis.X)

    @field_validator("axes", mode="before")
    def _cast_axes(cls, value: Any) -> tuple[Axis, ...]:
        return tuple(value)

    @model_validator(mode="after")
    def _validate_axes_space(cls, value: Any) -> Any:
        shape = getattr(value, "shape", ())
        scale = getattr(value, "scale", ())
        axes = getattr(value, "axes", ())
        ndim = len(shape)
        if len(scale) != ndim:
            raise ValueError(
                f"length of scale and shape must match ({len(scale)}, {ndim})"
            )
        if len(axes) < ndim:
            raise ValueError(f"Only {len(axes)} axes provided but got {ndim} dims")
        object.__setattr__(value, "axes", axes[-ndim:] if ndim else ())
        return value


class ShapeScaleSpace(_AxesSpace):
    shape: tuple[int, ...]
    scale: tuple[float, ...] = ()

    @model_validator(mode="before")
    def _cast_ob(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if "shape" not in value:
                raise ValueError("Must provide 'shape' in the input dictionary.")
            if not (scale := value.get("scale")):
                scale = 1
            if isinstance(scale, float | int):
                scale = (scale,) * len(value.get("shape", ()))
            value["scale"] = scale
        return value

    @computed_field
    def extent(self) -> tuple[float, ...]:
        return tuple(
            round(s * x, 12) for x, s in zip(self.shape, self.scale, strict=False)
        )


class ExtentScaleSpace(_AxesSpace):
    extent: tuple[float, ...]
    scale: tuple[float, ...]

    @model_validator(mode="before")
    def _cast_ob(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if "extent" not in value:
                raise ValueError("Must provide 'extent' in the input dictionary.")
            if "scale" not in value:
                raise ValueError("Must provide 'scale' in the input dictionary.")
            scale = value.get("scale")
            if isinstance(scale, float | int):
                scale = (scale,) * len(value.get("extent", ()))
            value["scale"] = scale
        return value

    @computed_field  # type: ignore
    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(x / s) for x, s in zip(self.extent, self.scale, strict=False))


class ShapeExtentSpace(_AxesSpace):
    shape: tuple[int, ...]
    extent: tuple[float, ...]

    @computed_field  # type: ignore
    @property
    def scale(self) -> tuple[float, ...]:
        return tuple(x / s for x, s in zip(self.extent, self.shape, strict=False))


ConcreteSpace = ExtentScaleSpace | ShapeExtentSpace | ShapeScaleSpace


class _RelativeSpace(_Space):
    reference: ConcreteSpace | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def scale(self) -> tuple[float, ...]:
        raise NotImplementedError

    @property
    def axes(self) -> tuple[str, ...]:
        if not self.reference:
            raise ValueError("Must provide a reference space.")
        return self.reference.axes


class DownscaledSpace(_RelativeSpace):
    downscale: tuple[int, ...] | int

    def rescale(self, img: xrDataArray) -> xrDataArray:
        if isinstance(self.downscale, int | float):
            axes = {ax: self.downscale for ax in self.axes}
        elif isinstance(self.downscale, Sequence):
            axes = dict(zip(self.axes, self.downscale, strict=False))

        return img.coarsen(axes).sum()  # type: ignore

    @computed_field  # type: ignore
    @property
    def shape(self) -> tuple[int, ...]:
        if not self.reference:
            raise ValueError("Must provide a reference space.")

        if isinstance(self.downscale, int | float):
            return tuple(int(x / self.downscale) for x in self.reference.shape)

        return tuple(
            int(x / d)
            for x, d in zip(self.reference.shape, self.downscale, strict=False)
        )

    @computed_field  # type: ignore
    @property
    def scale(self) -> tuple[float, ...]:
        if not self.reference:
            raise ValueError("Must provide a reference space.")

        if isinstance(self.downscale, int | float):
            return tuple(s * self.downscale for s in self.reference.scale)

        return tuple(
            s * d for s, d in zip(self.reference.scale, self.downscale, strict=False)
        )


class UpscaledSpace(_RelativeSpace):
    upscale: tuple[float, ...] | int

    @computed_field  # type: ignore
    @property
    def shape(self) -> tuple[int, ...]:
        if not self.reference:
            raise ValueError("Must provide a reference space.")

        if isinstance(self.upscale, int | float):
            return tuple(int(x * self.upscale) for x in self.reference.shape)

        return tuple(
            int(x * u) for x, u in zip(self.reference.shape, self.upscale, strict=False)
        )

    @computed_field  # type: ignore
    @property
    def scale(self) -> tuple[float, ...]:
        if not self.reference:
            raise ValueError("Must provide a reference space.")

        if isinstance(self.upscale, int | float):
            return tuple(s / self.upscale for s in self.reference.scale)

        return tuple(
            s / u for s, u in zip(self.reference.scale, self.upscale, strict=False)
        )


Space = ConcreteSpace | DownscaledSpace | UpscaledSpace
