"""The docs' interactive charts reimplement some of microsim's physics in JavaScript.

These tests run that JavaScript (docs/javascripts/physics/) in node and compare it
with microsim, so the two cannot drift apart unnoticed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from microsim import schema as ms
from microsim.schema.optical_config import lib

JS_TESTS = Path(__file__).parent / "js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _run(script: str, payload: dict) -> list[dict]:
    assert NODE is not None
    result = subprocess.run(
        [NODE, str(JS_TESTS / script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    # JS writes whole floats without a decimal point; keep them floats
    return json.loads(result.stdout, parse_int=float)  # type: ignore[no-any-return]


def _spec(spectrum: ms.Spectrum) -> dict:
    return {"w": spectrum.wavelength.tolist(), "v": spectrum.intensity.tolist()}


def _gauss(center: float, width: float, step: float = 1) -> ms.Spectrum:
    w = np.arange(center - 4 * width, center + 4 * width, step)
    return ms.Spectrum(wavelength=w, intensity=np.exp(-(((w - center) / width) ** 2)))


FLUOROPHORES = [
    ms.Fluorophore(
        name="green",
        excitation_spectrum=_gauss(488, 25),
        emission_spectrum=_gauss(510, 20),
        extinction_coefficient=56000,
        quantum_yield=0.6,
        lifetime_ns=2.6,
    ),
    # no lifetime (no saturation), no QY (treated as 1), half-nm emission grid
    ms.Fluorophore(
        name="red",
        excitation_spectrum=_gauss(580, 30),
        emission_spectrum=_gauss(610, 25, step=0.5),
        extinction_coefficient=72000,
    ),
]

CHANNELS = [
    lib.FITC,
    lib.TRITC,
    # two weighted light sources, and an inverted beam splitter
    ms.OpticalConfig(
        name="lights",
        lights=[
            ms.LightSource(spectrum=_gauss(480, 10), power=2),
            ms.LightSource.laser(561, power=1),
        ],
        filters=[
            ms.Bandpass(bandcenter=500, bandwidth=150, placement="EX"),
            ms.Longpass(cuton=530, placement="BSi"),
            ms.Bandpass(bandcenter=600, bandwidth=100, placement="EM"),
        ],
    ),
]

PARAMS = [
    {"power": None, "qe": None, "na": 1.4, "specimenRI": 1.47, "saturate": True},
    {"power": 5e4, "qe": 0.8, "na": 0.8, "specimenRI": 1.33, "saturate": True},
    {"power": 5e4, "qe": "spectrum", "na": 1.49, "specimenRI": 1.33, "saturate": False},
]
QE_SPECTRUM = _gauss(550, 150)


def _fluor_json(f: ms.Fluorophore) -> dict:
    return {
        "ec": f.extinction_coefficient,
        "qy": f.quantum_yield,
        "lifetime_ns": f.lifetime_ns,
        "ex": _spec(f.excitation_spectrum),
        "em": _spec(f.emission_spectrum),
    }


def _channel_json(oc: ms.OpticalConfig) -> dict:
    return {
        "filters": [
            {"placement": f.placement.value, "spectrum": _spec(f.spectrum)}
            for f in oc.filters
        ],
        "lights": [
            {"power": li.power, "spectrum": _spec(li.spectrum)} for li in oc.lights
        ],
    }


def _assert_same(js: dict, da: object) -> None:
    da = da.squeeze()  # type: ignore[attr-defined]
    np.testing.assert_array_equal(js["w"], da.coords["w"].values)
    np.testing.assert_allclose(js["v"], da.values, rtol=1e-9, atol=0)
    np.testing.assert_allclose(js["total"], float(da.sum()), rtol=1e-9)


def test_photometry() -> None:
    cases = []
    for p in PARAMS:
        qe = _spec(QE_SPECTRUM) if p["qe"] == "spectrum" else p["qe"]
        for fi in range(len(FLUOROPHORES)):
            for ci in range(len(CHANNELS)):
                cases.append({"fluor": fi, "channel": ci, "params": {**p, "qe": qe}})
    payload = {
        "fluorophores": [_fluor_json(f) for f in FLUOROPHORES],
        "channels": [_channel_json(oc) for oc in CHANNELS],
        "cases": cases,
    }
    results = _run("photometry.mjs", payload)

    for case, js in zip(cases, results, strict=True):
        p = case["params"]
        fluor = FLUOROPHORES[case["fluor"]]
        oc = CHANNELS[case["channel"]].model_copy(update={"power": p["power"]})
        qe = QE_SPECTRUM if isinstance(p["qe"], dict) else p["qe"]
        sat = p["saturate"]
        lens = ms.ObjectiveLens(numerical_aperture=p["na"], specimen_ri=p["specimenRI"])

        _assert_same(js["cross_section"], fluor.absorption_cross_section)
        _assert_same(js["illumination_flux_density"], oc.illumination_flux_density)
        _assert_same(js["absorbed"], oc.absorption_rate(fluor))
        _assert_same(js["emitted"], oc.total_emission_rate(fluor, saturate=sat))
        _assert_same(
            js["filtered"],
            oc.filtered_emission_rate(fluor, detector_qe=qe, saturate=sat),
        )
        assert js["collection"] == pytest.approx(lens.collection_efficiency, rel=1e-12)
        assert js["saturation"] == pytest.approx(
            oc.saturation_parameter(fluor), rel=1e-9
        )
