from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from ..models import Sample
from .utils._bresenham import drawlines_bresenham

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import NDArray

try:
    import cupy as xp
except ImportError:
    xp = np


class MatsLines(BaseModel, Sample):
    density: int = 1
    length: int = 10
    azimuth: int = 10
    max_r: float = 0.9

    def _gen_vertices(
        self, shape: tuple[int, ...], xypad: int = 1, zpad: int = 1
    ) -> tuple[NDArray, NDArray]:
        *nz, ny, nx = shape
        numlines = shape[-1] * self.density

        # random set of angles
        alpha = xp.random.rand(numlines) * 2 * np.pi
        if nz:
            alphaz = np.pi / 2 + xp.random.rand(numlines) * np.pi / self.azimuth
        else:
            alphaz = np.pi / 2

        # random set of x, y, z centers
        x1 = xp.random.randint(xypad, nx - xypad, size=numlines)
        y1 = xp.random.randint(xypad, ny - xypad, size=numlines)
        if nz:
            z1 = xp.random.randint(zpad, nz[0] - zpad, size=numlines)

        # find other end of line given alpha and length
        lens = nx / 20 + self.length * ny / 20 * xp.random.rand(numlines)
        x2 = xp.clip(
            xp.round(x1 + xp.sin(alphaz) * xp.cos(alpha) * lens), xypad, nx - xypad
        )
        y2 = xp.clip(
            xp.round(y1 + xp.sin(alphaz) * xp.sin(alpha) * lens), xypad, nx - xypad
        )

        if nz:
            z2 = xp.clip(np.round(z1 + np.cos(alphaz) * lens), zpad, nz[0] - zpad)
            return xp.stack([z1, y1, x1]).T, xp.stack([z2, y2, x2]).T
        return xp.stack([y1, x1]).T, xp.stack([y2, x2]).T

    def render(self, space: NDArray | xr.DataArray):
        start, end = self._gen_vertices(space.shape)
        c = xp.concatenate([start, end], axis=1).astype(np.int32)
        data = np.zeros(space.shape).astype(np.int32)
        # TODO: make bresenham work on GPU
        if hasattr(c, "get"):
            c = c.get()
        drawlines_bresenham(c, data, self.max_r)
        return space + data
