# Copyright (c) 2015, Warren Weckesser.  All rights reserved.
# This software is licensed according to the "BSD 2-clause" license.
from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def draw_line_2d(
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


def draw_line_3d(
    x0: int,
    y0: int,
    z0: int,
    x1: int,
    y1: int,
    z1: int,
    grid: np.ndarray,
    max_r: float,
    width: float = 1.0,
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

    if max_r < 0:
        max_r = sqrt(zr**2 + yr**2 + xr**2)

    while True:
        if width != 1:
            # Draw a sphere around the current point with the given width
            draw_sphere(grid, x0, y0, z0, width)
        else:
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


def draw_sphere(grid: np.ndarray, x0: int, y0: int, z0: int, radius: float) -> None:
    """Draw a sphere of a given radius around a point in a 3D grid."""
    z_range = range(int(max(0, z0 - radius)), int(min(grid.shape[0], z0 + radius + 1)))
    y_range = range(int(max(0, y0 - radius)), int(min(grid.shape[1], y0 + radius + 1)))
    x_range = range(int(max(0, x0 - radius)), int(min(grid.shape[2], x0 + radius + 1)))
    for z in z_range:
        for y in y_range:
            for x in x_range:
                distance = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2
                if distance <= radius**2:
                    grid[z, y, x] += 1


try:
    from numba import njit
except Exception:
    pass
else:
    draw_line_2d = njit(draw_line_2d)
    draw_line_3d = njit(draw_line_3d)
    draw_sphere = njit(draw_sphere)
