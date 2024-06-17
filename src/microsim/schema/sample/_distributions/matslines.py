from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from microsim.schema.backend import NumpyAPI
from microsim.schema.sample._distributions._base import _BaseDistribution

if TYPE_CHECKING:
    import numpy.typing as npt

    from microsim._data_array import xrDataArray


class MatsLines(_BaseDistribution):
    type: Literal["matslines"] = "matslines"
    density: float = 1
    length: int = 10
    azimuth: int = 10
    max_r: float = 0.9

    def cache_path(self) -> tuple[str, ...] | None:
        data = self.model_dump(mode="json").items()
        return ("matslines", *(f"{k}_{v}" for k, v in data if k != "type"))

    def _gen_vertices(
        self, xp: NumpyAPI, shape: tuple[int, ...], xypad: int = 1, zpad: int = 1
    ) -> tuple[npt.NDArray, npt.NDArray]:
        *nz, ny, nx = shape
        numlines = int(shape[-1] * self.density)

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

    def render(self, space: xrDataArray, xp: NumpyAPI | None = None) -> xrDataArray:
        xp = xp or NumpyAPI()

        start, end = self._gen_vertices(xp, space.shape)
        c = xp.concatenate([start, end], axis=1).astype(np.int32)
        data = np.zeros(space.shape).astype(np.int32)
        # TODO: make bresenham work on GPU
        if hasattr(c, "get"):
            c = c.get()
        drawlines_bresenham(c, data, self.max_r)
        # TODO: Multi-fluorophore setup: this addition should be replaced by setting
        # data in a specific dimension and index of space.
        return space + xp.asarray(data).astype(space.dtype)


def drawlines_bresenham(
    segments: npt.NDArray, grid: npt.NDArray, max_r: float = 2.0
) -> None:
    from microsim._draw import draw_line_2d, draw_line_3d

    if grid.ndim == 2:
        for segment in segments:
            y0, x0, y1, x1 = (int(x) for x in segment)
            draw_line_2d(x0, y0, x1, y1, grid, max_r)
    elif grid.ndim == 3:
        for segment in segments:
            z0, y0, x0, z1, y1, x1 = (int(x) for x in segment)
            draw_line_3d(x0, y0, z0, x1, y1, z1, grid, max_r)
    else:
        raise ValueError(f"grid must be either 2 or 3 dimensional.  Got {grid.ndim}")
