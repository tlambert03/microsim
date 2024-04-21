# Copyright (c) 2015, Warren Weckesser.  All rights reserved.
# This software is licensed according to the "BSD 2-clause" license.
from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def bres_draw_segment_2d(
    y0: int, x0: int, y1: int, x1: int, grid: np.ndarray, max_r: float
) -> None:
    """Bresenham's algorithm.

    See http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    """
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx + dy
    yr = grid.shape[0] / 2
    xr = grid.shape[1] / 2

    while True:
        if sqrt(((x0 - xr) / xr) ** 2 + ((y0 - yr) / yr) ** 2) <= max_r:
            grid[y0, x0] += 1

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def bres_draw_segment_3d(
    x0: int, y0: int, z0: int, x1: int, y1: int, z1: int, grid: np.ndarray, max_r: float
) -> None:
    """Bresenham's algorithm.

    See http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    sz = 1 if z0 < z1 else -1

    dm = max(dx, dy, dz)
    i: int = dm
    cx: float = dm / 2
    cy: float = dm / 2
    cz: float = dm / 2

    zr = grid.shape[0] / 2
    yr = grid.shape[1] / 2
    xr = grid.shape[2] / 2

    while True:
        r = ((x0 - xr) / xr) ** 2 + ((y0 - yr) / yr) ** 2 + ((z0 - zr) / zr) ** 2
        if sqrt(r) <= max_r:
            grid[z0, y0, x0] += 1
        if i == 0:
            break

        cx -= dx
        if cx < 0:
            cx += dm
            x0 += sx
        cy -= dy
        if cy < 0:
            cy += dm
            y0 += sy
        cz -= dz
        if cz < 0:
            cz += dm
            z0 += sz
        i -= 1


try:
    from numba import jit
except Exception:
    pass
else:
    bres_draw_segment_2d = jit(nopython=True)(bres_draw_segment_2d)
    bres_draw_segment_3d = jit(nopython=True)(bres_draw_segment_3d)
