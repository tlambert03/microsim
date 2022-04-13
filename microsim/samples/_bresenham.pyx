# Copyright (c) 2015, Warren Weckesser.  All rights reserved.
# This software is licensed according to the "BSD 2-clause" license.

cimport cython
from libc.math cimport abs, sqrt


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int bres_draw_segment_2d(int[:] coord, int[:, :] grid, float max_r):
    """Bresenham's algorithm.

    See http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    """
    cdef int x0, y0, x1, y1, dx, dy, sx, sy, err, e2

    y0 = coord[0]
    x0 = coord[1]
    y1 = coord[2]
    x1 = coord[3]

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

    return 0


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int bres_draw_segment_3d(int[:] coord, int[:, :, :] grid, float max_r):
    """Bresenham's algorithm.

    See http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    """
    cdef int x0, y0, z0, x1, y1, z1, dx, dy, dz, sx, sy, sz, dm, i
    cdef float zr, yr, xr

    z0 = coord[0]
    y0 = coord[1]
    x0 = coord[2]
    z1 = coord[3]
    y1 = coord[4]
    x1 = coord[5]

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    sz = 1 if z0 < z1 else -1

    dm = max(dx, dy, dz)
    i = dm
    x1 = y1 = z1 = dm/2

    zr = grid.shape[0] / 2
    yr = grid.shape[1] / 2
    xr = grid.shape[2] / 2

    while True:
        r = ((x0 - xr) / xr) ** 2 + ((y0 - yr) / yr) ** 2 + ((z0 - zr) / zr) ** 2
        if sqrt(r) <= max_r:
            grid[z0, y0, x0] += 1
        if i == 0:
            break

        x1 -= dx
        if (x1 < 0):
            x1 += dm
            x0 += sx
        y1 -= dy
        if (y1 < 0):
            y1 += dm
            y0 += sy
        z1 -= dz
        if (z1 < 0):
            z1 += dm
            z0 += sz
        i -= 1

    return 0


def drawlines_bresenham(segments, grid, max_r=2):
    if grid.ndim == 2:
        for segment in segments:
            bres_draw_segment_2d(segment, grid, max_r)
    elif grid.ndim == 3:
        for segment in segments:
            bres_draw_segment_3d(segment, grid, max_r)
    else:
        raise ValueError(f'grid must be either 2 or 3 dimensional.  Got {grid.ndim}')
