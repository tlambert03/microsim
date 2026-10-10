from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from microsim import schema as ms
from microsim.psf import make_confocal_psf
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


# flat spectra across the visible, so the fluorophore is excited and detected in FITC
_WAVES = np.arange(350, 750)
FLAT = ms.Fluorophore(
    name="flat",
    excitation_spectrum={"wavelength": _WAVES, "intensity": np.ones(_WAVES.size)},
    emission_spectrum={"wavelength": _WAVES, "intensity": np.ones(_WAVES.size)},
    extinction_coefficient=50_000,
)


@pytest.mark.parametrize("nz_out", [4, 8])
def test_widefield_plane_brightness_independent_of_stack_depth(nz_out: int) -> None:
    # every widefield exposure collects light from all fluorophores, so the photons
    # in one plane must not depend on how many planes are simulated
    sim = ms.Simulation(
        truth_space={"upscale": 2},
        output_space=ms.ShapeScaleSpace(shape=(nz_out, 32, 32), scale=(0.2, 0.1, 0.1)),
        sample=[
            ms.FluorophoreDistribution(
                distribution=ms.MatsLines(density=0.5, length=10, max_r=0.5),
                fluorophore=FLAT,
            )
        ],
        modality=ms.Widefield(),
        settings=ms.Settings(random_seed=0),
    )
    assert sim.ground_truth().sum() > 0  # small samples can be empty by chance
    img = sim.digital_image(with_detector_noise=False, exposure_ms=1000)
    per_fluor = float(img[0, nz_out // 2].sum()) / sim.ground_truth().sum().item()
    rate = float(sim.filtered_emission_rates().sum())
    assert rate > 0
    assert per_fluor == pytest.approx(rate, rel=0.15)
