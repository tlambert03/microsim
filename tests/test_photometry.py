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
    # NA 1.4 in oil (n=1.515): half angle ~67.5 deg -> (1 - cos) / 2 ~ 0.31
    assert ObjectiveLens(numerical_aperture=1.4).collection_efficiency == pytest.approx(
        0.309, abs=1e-3
    )
    assert ObjectiveLens(numerical_aperture=0.5).collection_efficiency < 0.03


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
