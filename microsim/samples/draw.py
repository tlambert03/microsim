from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    import cupy as xp
except ImportError:
    xp = np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _bresenhamline_nslope(slope: NDArray):
    """
    Normalize slope for Bresenham's line algorithm.

    >>> s = np.array([[-2, -2, -2, 0]])
    >>> _bresenhamline_nslope(s)
    array([[-1., -1., -1.,  0.]])

    >>> s = np.array([[0, 0, 0, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  0.,  0.]])

    >>> s = np.array([[0, 0, 9, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  1.,  0.]])
    """

    scale = xp.amax(xp.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = xp.ones(1)
    normalizedslope = xp.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = xp.zeros(slope[0].shape)
    return normalizedslope


def line_bresenham(
    start: NDArray, end: NDArray, max_iter: int = -1, include_start=True
):
    """Return list of points from (start, end) by ray tracing a line b/w the points.

    Parameters
    ----------
    start : NDArray
        An array of start points (number of points x dimension)
    end : NDArray
        An end point (1 x dimension) or An array of end points corresponding to each
        start point (number of points x dimension)
    max_iter : int, optional
        Max points to traverse. If -1, maximum number of required points are traversed,
        by default -1.
    include_start : bool
        Whether to include starting coordinates in line.

    Returns
    -------
    np.ndarray
        linevox (n x dimension) A cumulative array of all points traversed by all the
        lines so far.

    Examples
    --------
    >>> start = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> end = np.zeros(start.shape[1])
    >>> line_bresenham(start, end)
    array([[ 3,  1,  9,  0]
           [ 3,  1,  8,  0],
           [ 2,  1,  7,  0],
           [ 2,  1,  6,  0],
           [ 2,  1,  5,  0],
           [ 1,  0,  4,  0],
           [ 1,  0,  3,  0],
           [ 1,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0, -1,  0],
           [ 0,  0, -2,  0],
           [ 0,  0, -3,  0],
           [ 0,  0, -4,  0],
           [ 0,  0, -5,  0],
           [ 0,  0, -6,  0]])

    """
    # Return the points as a single array
    start = xp.asarray(start).astype(int)
    end = xp.asarray(end).astype(int)
    if max_iter == -1:
        max_iter = xp.amax(xp.amax(xp.abs(end - start), axis=1))

    ndim = start.shape[-1]
    nslope = _bresenhamline_nslope(end - start)

    # steps to iterate on
    stepseq = xp.arange(1, max_iter + 1)
    stepmat: NDArray = xp.tile(stepseq, (ndim, 1)).T

    # some hacks for broadcasting properly
    bline = start[:, xp.newaxis, :] + nslope[:, xp.newaxis, :] * stepmat
    if include_start:
        bline = xp.hstack([start[:, xp.newaxis, :], bline])
    # Approximate to nearest int
    lines = xp.array(xp.rint(bline), dtype=start.dtype)
    return lines.reshape(-1, start.shape[-1])
