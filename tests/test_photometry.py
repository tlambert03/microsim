from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from microsim import schema as ms
from microsim.illum._spinning_disc import pinhole_mask
from microsim.psf import make_confocal_psf, make_spinning_disk_psf
from microsim.schema.lens import ObjectiveLens
from microsim.schema.optical_config.lib import FITC
from tests._util import skipif_no_internet


def test_objective_numerical_aperture() -> None:
    lens = ObjectiveLens(numerical_aperture=0.8)
    assert ObjectiveLens.model_validate(lens.model_dump()) == lens
    with pytest.raises(ValidationError, match="cannot be greater"):
        ObjectiveLens(numerical_aperture=1.6)
    with pytest.raises(ValidationError, match="immersion_medium_ri "):
        ObjectiveLens(numerical_aperture=1.4, immersion_medium_ri=1.33)
    with pytest.raises(ValidationError, match="cannot be greater"):
        lens.immersion_medium_ri = 0.7  # validate_assignment


def test_deprecated_na() -> None:
    with pytest.warns(FutureWarning, match="'na' has been renamed") as record:
        sim = ms.Simulation(
            truth_space={"shape": (4, 4, 4), "scale": (1, 1, 1)},
            sample=[],
            objective_lens={"na": 0.5},
        )
    assert sim.objective_lens.numerical_aperture == 0.5
    assert record[0].filename == __file__  # points at user code
    with pytest.warns(FutureWarning), pytest.raises(ValidationError, match="cannot be"):
        ObjectiveLens(na=1.6)
    with pytest.raises(ValidationError, match="Cannot specify both"):
        ObjectiveLens(na=0.5, numerical_aperture=0.5)


def test_unknown_fields_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        ObjectiveLens(working_distance=100)
    with pytest.raises(ValidationError, match="Extra inputs"):
        ms.Simulation(
            truth_space={"shape": (8, 8, 8), "scale": (1, 1, 1)},
            sample=[],
            output="out.tif",
        )


def test_computed_fields_round_trip() -> None:
    space = ms.ShapeScaleSpace(shape=(4, 8, 8), scale=(0.1, 0.1, 0.1))
    data = space.model_dump()
    assert "extent" in data
    assert ms.ShapeScaleSpace.model_validate(data) == space
    data["extent"] = (1, 1, 1)
    with pytest.raises(ValidationError, match="'extent' is computed"):
        ms.ShapeScaleSpace.model_validate(data)


def test_collection_efficiency() -> None:
    # NA 1.4 into specimen n=1.47: theta ~72 deg -> (1 - cos) / 2 ~ 0.347
    lens = ObjectiveLens(numerical_aperture=1.4)
    assert lens.collection_efficiency == pytest.approx(0.347, abs=1e-3)
    assert ObjectiveLens(numerical_aperture=0.5).collection_efficiency < 0.03
    # NA > specimen RI (e.g. oil objective into water): full hemisphere
    water = ObjectiveLens(numerical_aperture=1.4, specimen_ri=1.33)
    assert water.collection_efficiency == pytest.approx(0.5)


@skipif_no_internet
def test_excitation_saturation() -> None:
    egfp = ms.Fluorophore.from_fpbase("EGFP")
    linear = egfp.model_copy(update={"lifetime_ns": None})
    rates = {}
    for power in (1, 1e6):
        oc = FITC.model_copy(update={"power": power})
        rates[power] = (
            oc.total_emission_rate(egfp).sum().item(),
            oc.total_emission_rate(linear).sum().item(),
        )
    # negligible saturation at widefield irradiance
    assert rates[1][0] == pytest.approx(rates[1][1], rel=1e-4)
    # strong saturation at confocal irradiance, capped at QY / lifetime
    sat, lin = rates[1e6]
    k = lin / egfp.quantum_yield  # absorption rate
    assert sat == pytest.approx(lin / (1 + k * egfp.lifetime_ns * 1e-9))
    assert sat < egfp.quantum_yield / (egfp.lifetime_ns * 1e-9)


def test_confocal_psf_pinhole_throughput() -> None:
    kw = {"nz": 9, "nx": 65, "dz": 0.1, "dxy": 0.03, "normalize": False}
    peaks = [make_confocal_psf(pinhole_au=au, **kw).max() for au in (0.25, 1, 4)]
    assert peaks[0] < peaks[1] < peaks[2] <= 1
    # 1 AU pinhole passes most of the in-focus emission
    assert 0.6 < peaks[1] < 0.95
    assert np.isclose(make_confocal_psf(**{**kw, "normalize": "sum"}).sum(), 1)


def test_confocal_psf_saturation() -> None:
    kw = {"nz": 9, "nx": 65, "dz": 0.1, "dxy": 0.03, "normalize": False}
    linear = make_confocal_psf(**kw)
    s = 10.0
    sat = make_confocal_psf(**kw, saturation=s)
    # peak reduced by exactly 1/(1+s); off-focus regions are reduced less
    assert sat.max() == pytest.approx(linear.max() / (1 + s), rel=1e-6)
    assert (sat / sat.max()).sum() > (linear / linear.max()).sum()


def _green_fluor() -> ms.Fluorophore:
    wvl = np.arange(400, 650)
    return ms.Fluorophore(
        name="green",
        excitation_spectrum={
            "wavelength": wvl,
            "intensity": np.exp(-(((wvl - 488) / 20) ** 2)),
        },
        emission_spectrum={
            "wavelength": wvl,
            "intensity": np.exp(-(((wvl - 510) / 20) ** 2)),
        },
        extinction_coefficient=55_000,
        quantum_yield=0.6,
        lifetime_ns=2.6,
    )


def test_confocal_saturation_in_psf_not_rates() -> None:
    fluor = _green_fluor()
    oc = FITC.model_copy(update={"power": 1e7})
    s = oc.saturation_parameter(fluor)
    assert s > 1
    assert oc.saturation_parameter(fluor.model_copy(update={"lifetime_ns": None})) == 0

    kw = {
        "truth_space": {"shape": (4, 4, 4), "scale": (1, 1, 1)},
        "sample": [{"distribution": ms.MatsLines(), "fluorophore": fluor}],
        "channels": [oc],
    }
    wf = ms.Simulation(**kw, modality=ms.Widefield())
    cf = ms.Simulation(**kw, modality=ms.Confocal())

    # confocal emission rates are unsaturated; widefield rates are saturated
    cf_rates = cf.filtered_emission_rates()
    np.testing.assert_allclose(
        wf.filtered_emission_rates(), cf_rates / (1 + s), atol=1e-300
    )

    # confocal recovers the same saturation parameter from the rate coords
    em_spectrum = cf_rates.isel(c=0, f=0)
    assert cf.modality._saturation_parameter(em_spectrum) == pytest.approx(s, rel=1e-3)
    assert wf.modality._saturation_parameter(em_spectrum) == 0


def test_spinning_disk_psf() -> None:
    # window must span the defocused PSF (~10 um at 5 um defocus) to see crosstalk
    nx, dxy = 257, 0.1
    kw = {"nz": 11, "dz": 1.0, "nx": nx, "dxy": dxy, "em_wvl_um": 0.52}
    kw["ex_wvl_um"] = 0.52
    grid = np.hypot(*np.meshgrid(*[np.arange(nx) - nx // 2] * 2))
    nipkow = pinhole_mask(nx=nx, dxy_um=dxy, magnification=100)
    single = (grid <= 0.25 / dxy).astype(float)  # one 50 um pinhole at 100x
    focus, far = 5, 0  # z index of focal plane, and 5 um defocus

    # wide-open pinhole: a uniform thin plane at focus gets the time-averaged
    # irradiance everywhere and all collected light is detected (= widefield)
    open_ = make_spinning_disk_psf(
        pinhole_mask=np.ones((nx, nx)), **kw, pinhole_spacing_um=2.53
    )
    assert open_[focus].sum() == pytest.approx(1, abs=0.02)

    # crosstalk: same in-focus signal, but out-of-focus background plateaus near the
    # fill factor (pinhole area / pitch**2) instead of falling off
    xtalk = make_spinning_disk_psf(pinhole_mask=nipkow, pinhole_spacing_um=2.53, **kw)
    alone = make_spinning_disk_psf(pinhole_mask=single, pinhole_spacing_um=2.53, **kw)
    assert xtalk[focus].sum() == pytest.approx(alone[focus].sum(), rel=0.02)
    fill = np.pi * 25**2 / 253**2
    assert alone[far].sum() < 0.1 * fill
    assert xtalk[far].sum() == pytest.approx(fill, rel=0.3)

    # pinhole pitch cancels out of the signal, except through saturation
    wide = make_spinning_disk_psf(pinhole_mask=single, pinhole_spacing_um=5, **kw)
    np.testing.assert_allclose(wide, alone)
    sat = {
        sp: make_spinning_disk_psf(
            pinhole_mask=single, pinhole_spacing_um=sp, saturation=1e-3, **kw
        )
        for sp in (2.53, 5)
    }
    assert sat[5].max() < sat[2.53].max() < alone.max()


def test_spinning_disk_simulation() -> None:
    sim = ms.Simulation(
        truth_space={"shape": (4, 4, 4), "scale": (1, 1, 1)},
        sample=[{"distribution": ms.MatsLines(), "fluorophore": _green_fluor()}],
        modality={"type": "spinning_disk"},
    )
    assert isinstance(sim.modality, ms.SpinningDiskConfocal)
    # camera: sum in xy when downscaling, but each z-plane is one measurement
    assert sim.modality.rescale_mean_axes == ("z",)
    assert sim.modality.local_saturation  # saturation applied in the PSF


def test_spinning_disk_underfilled_excitation() -> None:
    # low-NA excitation beamlets stay localized out of focus, so a defocused point
    # images as an array of pinhole spots rather than a smooth blur
    nx, dxy = 161, 0.08
    kw = {"nz": 7, "dz": 1.0, "nx": nx, "dxy": dxy, "em_wvl_um": 0.52}
    kw.update(ex_wvl_um=0.488, pinhole_spacing_um=2.53)
    kw["pinhole_mask"] = pinhole_mask(nx=nx, dxy_um=dxy, magnification=100)
    c, pitch = nx // 2, round(2.53 / dxy)  # tangential neighbor at (c, c + pitch)

    def contrast(psf: np.ndarray) -> float:
        plane = psf[0]  # 3 um defocus
        return float(plane[c, c + pitch] / plane[c, c + pitch // 2])

    assert contrast(make_spinning_disk_psf(**kw, excitation_na=0.3)) > 3
    assert contrast(make_spinning_disk_psf(**kw)) < 1.5  # full NA 1.4: smooth
