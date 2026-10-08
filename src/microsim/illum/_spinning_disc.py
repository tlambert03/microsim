from functools import lru_cache

import numpy as np
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree


@lru_cache
def pinhole_coords(
    radii: tuple[float, float] = (15, 25),
    pinhole_spacing: float = 0.253,
    frame_per_rev: int = 12,
    spiral_spacing: float | None = None,
) -> np.ndarray:
    """Return (N, 2) pinhole (x, y) centers on a Yokogawa-style Nipkow disk.

    Pinholes lie on `frame_per_rev` interleaved Archimedean spirals, with equal pitch
    along each spiral (`pinhole_spacing`) and between adjacent spirals
    (`spiral_spacing`, default `pinhole_spacing`), so that pinhole density (and
    illumination) is uniform from the inner to the outer radius.  Each image frame is
    scanned by a rotation of `360 / frame_per_rev` degrees.  See Tanaami & Mikuriya
    (Yokogawa), EP0539691 / US5428475.

    Units are those of the inputs (defaults: mm, for a CSU-X1: 50 um pinholes at
    ~5x spacing, ~20,000 pinholes, ~1,000 in the 10 x 7 mm image area).
    """
    r0, r1 = radii
    pitch = pinhole_spacing
    # radial growth per radian of each spiral: r = r0 + b * theta, which advances by
    # `frame_per_rev` spiral spacings per turn (the interleaved spirals fill the gap)
    b = (spiral_spacing or pitch) * frame_per_rev / (2 * np.pi)
    # pinhole i is at arc length i * pitch, with arc length ~ r0*theta + b*theta**2/2
    # (neglects the radial term: relative error (b/r)**2/2 <= 5e-4 for the CSU-X1;
    # EP0539691 eq. 7)
    theta_max = (r1 - r0) / b
    n = int((r0 * theta_max + b * theta_max**2 / 2) / pitch) + 1
    theta = (np.sqrt(r0**2 + 2 * b * pitch * np.arange(n)) - r0) / b
    r = r0 + b * theta
    # interleaved spirals: copies rotated by 360 / frame_per_rev degrees
    phi = theta + 2 * np.pi * np.arange(frame_per_rev)[:, None] / frame_per_rev
    return np.column_stack([(r * np.cos(phi)).ravel(), (r * np.sin(phi)).ravel()])


@lru_cache
def pinhole_mask(
    nx: int,
    dxy_um: float,
    magnification: float,
    pinhole_diameter_um: float = 50,
    pinhole_spacing_um: float = 253,
    disk_radii_mm: tuple[float, float] = (15, 25),
    frames_per_rev: float = 12,
    image_size_mm: tuple[float, float] = (10, 7),
    n_rotations: int = 12,
) -> np.ndarray:
    """Time-averaged Nipkow-disk pinhole transmission around a pinhole.

    Returns an (nx, nx) mask in the sample plane (pixel size `dxy_um`), centered on a
    pinhole: the pinhole itself plus its neighbors on the disk, averaged over all
    pinholes in the image area (`image_size_mm`, centered between `disk_radii_mm`)
    and over the disk rotation of one frame.  Disk-plane sizes are divided by
    `magnification` (sample -> disk).  Used to model pinhole crosstalk.
    """
    pts = pinhole_coords(disk_radii_mm, pinhole_spacing_um * 1e-3, int(frames_per_rev))
    # reference pinholes: those inside the image area (tangential x radial), which
    # is assumed to be centered on the +x axis, midway between the disk radii
    r_center = sum(disk_radii_mm) / 2
    width, height = image_size_mm
    in_image = (np.abs(pts[:, 1]) <= width / 2) & (
        np.abs(pts[:, 0] - r_center) <= height / 2
    )
    refs = pts[in_image]
    # neighbors within the (diagonal) half-width of the window, in disk mm
    half = (nx // 2 + 1) * dxy_um * magnification * 1e-3 * np.sqrt(2)
    nbrs = cKDTree(pts).query_ball_point(refs, half)
    offsets = np.concatenate([pts[n] - r for n, r in zip(nbrs, refs, strict=True)])
    offsets *= 1e3 / magnification / dxy_um  # disk mm -> sample pixels

    # average over the rotation of the disk during one frame, then splat the pinhole
    # centers onto the grid (bilinear), weighted per reference pinhole
    angles = np.linspace(0, 2 * np.pi / frames_per_rev, n_rotations, endpoint=False)
    cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
    # camera axes: x tangential (disk y), y radial (disk x)
    x = (cos * offsets[:, 1] + sin * offsets[:, 0]).ravel() + nx // 2
    y = (-sin * offsets[:, 1] + cos * offsets[:, 0]).ravel() + nx // 2
    weight = 1 / (len(refs) * n_rotations)
    density = np.zeros((nx, nx))
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    for dy, dx, w in (
        (0, 0, (1 - fy) * (1 - fx)),
        (0, 1, (1 - fy) * fx),
        (1, 0, fy * (1 - fx)),
        (1, 1, fy * fx),
    ):
        yi, xi = y0 + dy, x0 + dx
        ok = (yi >= 0) & (yi < nx) & (xi >= 0) & (xi < nx)
        np.add.at(density, (yi[ok], xi[ok]), w[ok] * weight)

    # convolve pinhole centers with the (projected) pinhole aperture
    radius_px = pinhole_diameter_um / 2 / magnification / dxy_um
    grid = np.arange(nx) - nx // 2
    aperture = (np.hypot(*np.meshgrid(grid, grid)) <= radius_px).astype(float)
    mask: np.ndarray = np.clip(fftconvolve(density, aperture, mode="same"), 0, None)
    return mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage.draw import disk

    points = (pinhole_coords() * 100).astype(int) + 2500
    arr = np.zeros((5010, 5010))
    for p in points:
        arr[disk(p, 5)] = 1

    plt.imshow(arr)
    plt.show()
