from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union

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
        self, shape: Tuple[int, ...], xypad: int = 1, zpad: int = 1
    ) -> Tuple[NDArray, NDArray]:
        *nz, ny, nx = shape
        numlines = shape[-1] * self.density

        # random set of angles
        alpha = xp.random.rand(numlines) * 2 * np.pi
        if nz:
            alphaz = np.pi / 2 + xp.random.rand(numlines) * np.pi / self.azimuth
        else:
            alphaz = np.pi / 2

        # random set of x, y, z centers
        x1 = np.random.randint(xypad, nx - xypad, size=numlines)
        y1 = np.random.randint(xypad, ny - xypad, size=numlines)
        if nz:
            z1 = np.random.randint(zpad, nz[0] - zpad, size=numlines)

        # find other end of line given alpha and length
        lens = nx / 20 + self.length * ny / 20 * xp.random.rand(numlines)
        x2 = xp.clip(
            np.round(x1 + np.sin(alphaz) * np.cos(alpha) * lens), xypad, nx - xypad
        )
        y2 = xp.clip(
            np.round(y1 + np.sin(alphaz) * np.sin(alpha) * lens), xypad, nx - xypad
        )

        if nz:
            z2 = xp.clip(np.round(z1 + np.cos(alphaz) * lens), zpad, nz[0] - zpad)
            return xp.stack([z1, y1, x1]).T, xp.stack([z2, y2, x2]).T
        return xp.stack([y1, x1]).T, xp.stack([y2, x2]).T

    def render(self, space: Union[NDArray, xr.DataArray]):
        start, end = self._gen_vertices(space.shape)
        data = xp.zeros_like(space).astype(np.int32)
        c = xp.concatenate([start, end], axis=1).astype(np.int32)
        drawlines_bresenham(c, data, self.max_r)
        return space + data
