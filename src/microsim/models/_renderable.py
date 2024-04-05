from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import xarray as xr


class Renderable(Protocol):
    def render(self, space: xr.DataArray) -> None: ...

    def __mul__(self, other: Any) -> Renderable: ...

    def __sum__(self, other: Any) -> Renderable: ...
