from __future__ import annotations

import numpy as np
import pytest

from microsim import schema as ms
from microsim.psf import make_psf
from microsim.schema.modality._optics import CameraDetection, PinholeDetection
from microsim.schema.optical_config.lib import FITC

NA1_4 = ms.ObjectiveLens(numerical_aperture=1.4)
PSF_KW = {"dx": 0.05, "dz": 0.1, "objective": NA1_4, "em_wvl_nm": 520}


@pytest.mark.parametrize("pinhole_au", [None, 1], ids=["widefield", "confocal"])
def test_make_psf_odd_and_centered(pinhole_au: float | None) -> None:
    """PSFs have odd sizes (rounded up), with the peak on the center pixel."""
    for n, expect in [(17, 17), (16, 17)]:
        psf = make_psf(nz=n, nx=n, pinhole_au=pinhole_au, **PSF_KW)
        assert psf.shape == (expect, expect, expect)
        assert np.unravel_index(np.argmax(psf), psf.shape) == (expect // 2,) * 3


@pytest.mark.parametrize("pinhole_au", [None, 1], ids=["widefield", "confocal"])
def test_make_psf_normalization_independent_of_size(pinhole_au: float | None) -> None:
    """A laterally truncated PSF is not brightened to make up for the lost light."""
    small = make_psf(nz=9, nx=15, pinhole_au=pinhole_au, **PSF_KW)
    large = make_psf(nz=9, nx=101, pinhole_au=pinhole_au, **PSF_KW)
    np.testing.assert_allclose(small.max(), large.max(), rtol=1e-6)
    assert small[4].sum() < large[4].sum()
    if pinhole_au is None:  # widefield planes are fractions of the collected light
        assert large[4].sum() <= 1


@pytest.mark.parametrize(
    "modality", [ms.Widefield(), ms.Confocal()], ids=lambda m: m.type
)
def test_point_source_is_not_shifted(modality: ms.Modality) -> None:
    truth = np.zeros((16, 48, 48))
    truth[8, 24, 24] = 1000
    for radius in (3, None):  # odd sizes, whether or not the radius is capped
        sim = ms.Simulation.from_ground_truth(
            truth,
            scale=(0.05, 0.02, 0.02),
            modality=modality,
            channels=[FITC],
            settings=ms.Settings(max_psf_radius_aus=radius, cache=False),
        )
        img = np.asarray(sim.optical_image().data[0])
        assert np.unravel_index(np.argmax(img), img.shape) == (8, 24, 24)
        # laterally symmetric around the source
        profile = img[8, 24, 1:]  # pixels 1..47, centered on 24
        np.testing.assert_allclose(profile, profile[::-1], atol=1e-5 * profile.max())


def test_max_psf_radius_aus() -> None:
    """A smaller radius truncates the PSF (and auto grows with depth for widefield)."""

    def _image(radius: float | str | None) -> np.ndarray:
        sim = ms.Simulation(
            truth_space=ms.ShapeScaleSpace(shape=(8, 96, 96), scale=(0.05, 0.02, 0.02)),
            sample=ms.Sample(
                labels=[ms.FluorophoreDistribution(distribution=ms.MatsLines())]
            ),
            objective_lens=NA1_4,
            settings=ms.Settings(random_seed=1, max_psf_radius_aus=radius, cache=False),
        )
        return np.asarray(sim.optical_image().data)

    full = _image(None)
    assert not np.allclose(_image(2), full, rtol=0.01)  # 2 AU = ~9 px << 48 px
    np.testing.assert_allclose(_image(100), full)  # never larger than the volume

    camera, pinhole = CameraDetection(), PinholeDetection(radius_um=0.2)
    assert camera.psf_radius_aus(5, 0.52, NA1_4) > camera.psf_radius_aus(1, 0.52, NA1_4)
    assert pinhole.psf_radius_aus(5, 0.52, NA1_4) == 8
