from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from microsim.illum._spinning_disc import pinhole_coords, pinhole_mask


def test_pinhole_coords() -> None:
    # CSU-X1 defaults (mm): ~20,000 pinholes, ~1,000 in the 10 x 7 mm image area
    pts = pinhole_coords()
    assert 19_000 < len(pts) < 21_000
    r = np.hypot(*pts.T)
    assert r.min() >= 15 - 1e-9 and r.max() <= 25 + 1e-9
    in_image = (np.abs(pts[:, 1]) < 5) & (np.abs(pts[:, 0] - 20) < 3.5)
    assert 900 < in_image.sum() < 1200

    # equal pitch along and between spirals: nearest neighbors at ~one pitch
    dist, _ = cKDTree(pts).query(pts, k=2)
    np.testing.assert_allclose(dist[:, 1], 0.253, rtol=0.01)
    # uniform density from inner to outer radius (1 / pitch**2)
    for lo in (15.5, 23.5):
        ring = (r > lo) & (r < lo + 1)
        density = ring.sum() / (np.pi * ((lo + 1) ** 2 - lo**2))
        assert abs(density * 0.253**2 - 1) < 0.02


def test_pinhole_mask_fill_factor() -> None:
    # time-averaged transmission ~ pinhole area / pitch**2 (50 um pinholes, 253 pitch)
    # (window must span many pitches for the mean to converge)
    mask = pinhole_mask(nx=1025, dxy_um=0.04, magnification=100)
    assert mask[512, 512] > 0.99
    np.testing.assert_allclose(mask.mean(), np.pi * 25**2 / 253**2, rtol=0.05)
