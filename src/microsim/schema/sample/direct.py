from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from microsim.schema._base_model import SimBaseModel
from microsim.schema.backend import NumpyAPI

if TYPE_CHECKING:
    from microsim._data_array import DataArray


class FixedArrayTruth(SimBaseModel):
    type: Literal["fixed-array"] = "fixed-array"
    array: Any

    def render(self, space: DataArray, xp: NumpyAPI | None = None) -> DataArray:
        if space.shape != self.array.shape:
            raise ValueError(
                "This GroundTruth may only be used with simulation space of shape: "
                f"{self.array.shape}. Got: {space.shape}"
            )

        xp = xp or NumpyAPI()
        return space + xp.asarray(self.array).astype(space.dtype)
