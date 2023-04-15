from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr


class Renderable:
    def render(self, space: xr.DataArray):
        ...

    def __mul__(self, other) -> Renderable:
        ...

    def __sum__(self, other) -> Renderable:
        ...
