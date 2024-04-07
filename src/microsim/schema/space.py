from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    ValidatorFunctionWrapHandler,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_core import CoreSchema, core_schema


class FloatArray(npt.NDArray[np.floating]):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        list_schema = core_schema.list_schema(core_schema.float_schema())
        return core_schema.no_info_wrap_validator_function(
            cls._wrap_validator, list_schema
        )

    @classmethod
    def _wrap_validator(self, value, handler: ValidatorFunctionWrapHandler):
        return np.array(value, dtype=np.float64)


class _Space(BaseModel):
    if TYPE_CHECKING:
        shape: tuple[int, ...]
        scale: tuple[float, ...]

    def rescale(self, img: np.ndarray) -> np.ndarray:
        from microsim.util import downsample

        return downsample(img, self.downscale)

    def create(self, array_creator: Callable[[], npt.ArrayLike] = np.zeros):
        from microsim.util import uniformly_spaced_xarray

        return uniformly_spaced_xarray(
            shape=self.shape, scale=self.scale, array_creator=array_creator
        )


class CoordsSpace(_Space):
    coords: Mapping[str, FloatArray]


class _AxesSpace(_Space):
    axes: tuple[str, ...] = ("T", "C", "Z", "Y", "X")

    @field_validator("axes", mode="before")
    def _cast_axes(cls, value: Any):
        return tuple(value)

    @model_validator(mode="after")
    def _validate_axes_space(cls, value: Any):
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
        value.axes = axes[-ndim:] if ndim else ()
        return value

    @computed_field(repr=False)
    def coords(self) -> Mapping[str, FloatArray]:
        return {
            ax: np.arange(sh) * sc
            for ax, sh, sc in zip(self.axes, self.shape, self.scale)
        }


class ShapeScaleSpace(_AxesSpace):
    shape: tuple[int, ...]
    scale: tuple[float, ...] = ()

    @model_validator(mode="before")
    def _cast_ob(cls, value: Any):
        if isinstance(value, dict):
            if "shape" not in value:
                raise ValueError("Must provide 'shape' in the input dictionary.")
            if not (scale := value.get("scale")):
                scale = 1
            if isinstance(scale, (float, int)):
                scale = (scale,) * len(value.get("shape", ()))
            value["scale"] = scale
        return value

    @computed_field
    def extent(self) -> tuple[float, ...]:
        return tuple(round(s * x, 12) for x, s in zip(self.shape, self.scale))


class ExtentScaleSpace(_AxesSpace):
    extent: tuple[float, ...]
    scale: tuple[float, ...]

    @model_validator(mode="before")
    def _cast_ob(cls, value: Any):
        if isinstance(value, dict):
            if "extent" not in value:
                raise ValueError("Must provide 'extent' in the input dictionary.")
            if "scale" not in value:
                raise ValueError("Must provide 'scale' in the input dictionary.")
            scale = value.get("scale")
            if isinstance(scale, (float, int)):
                scale = (scale,) * len(value.get("extent", ()))
            value["scale"] = scale
        return value

    @computed_field
    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(x / s) for x, s in zip(self.extent, self.scale))


class ShapeExtentSpace(_AxesSpace):
    shape: tuple[int, ...]
    extent: tuple[float, ...]

    @computed_field
    @property
    def scale(self) -> tuple[float, ...]:
        return tuple(x / s for x, s in zip(self.extent, self.shape))


ConcreteSpace = ExtentScaleSpace | ShapeExtentSpace | ShapeScaleSpace


class _RelativeSpace(_Space):
    reference: ConcreteSpace | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError


class DownscaledSpace(_RelativeSpace):
    downscale: tuple[float, ...] | int

    @computed_field
    def shape(self) -> tuple[int, ...]:
        if not self.reference:
            raise ValueError("Must provide a reference space.")

        if isinstance(self.downscale, (int, float)):
            return tuple(int(x / self.downscale) for x in self.reference.shape)

        return tuple(int(x / d) for x, d in zip(self.reference.shape, self.downscale))


class UpscaledSpace(_RelativeSpace):
    upscale: tuple[float, ...] | int

    @computed_field
    def shape(self) -> tuple[int, ...]:
        if not self.reference:
            raise ValueError("Must provide a reference space.")

        if isinstance(self.upscale, (int, float)):
            return tuple(int(x * self.upscale) for x in self.reference.shape)

        return tuple(int(x * u) for x, u in zip(self.reference.shape, self.upscale))


Space = ConcreteSpace | DownscaledSpace | UpscaledSpace
