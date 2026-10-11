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
from microsim._data_array import DataArray
from microsim.schema.backend import NumpyAPI
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


def _measured_snr(detector: ms.Detector, electrons: float) -> float:
    """SNR of a flat image recorded by `detector`, in a 1 s exposure."""
    xp = NumpyAPI()
    xp.set_random_seed(0)
    image = DataArray(np.full((1, 1, 400, 400), electrons), dims=("c", "z", "y", "x"))
    gray = np.asarray(detector.render(image, exposure_ms=1000, xp=xp), dtype=float)
    signal = (gray - detector.offset) * detector.conversion_factor
    return float(signal.mean() / signal.std())


# no clipping or quantization, and an offset well above the read noise
_EXACT = {
    "bit_depth": 32,
    "electrons_per_adu": 0.01,
    "offset": 10**6,
    "dark_current": 0,
}
SNR_CASES = [
    (ms.CameraCCD(read_noise=6, **_EXACT), {"readNoise": 6}),
    (
        ms.CameraCMOS(read_noise=1.5, averaging=4, **_EXACT),
        {"readNoise": 1.5, "averaging": 4},
    ),
    (
        ms.CameraEMCCD(em_gain=300, read_noise=50, full_well=10**9, **_EXACT),
        {
            "readNoise": 50,
            "gain": 300,
            "enf": ms.CameraEMCCD(em_gain=300).excess_noise_factor,
        },
    ),
]


def test_detectors() -> None:
    emccd = [(g, n) for g in (1, 1.5, 10, 300, 1000) for n in (604, 100, 1)]
    pmt = [
        {"hv": hv, "ref_hv": 0.8, "ref_gain": 5e5, "hv_exponent": e, "dynode_stages": d}
        for hv in (0.5, 0.8, 0.9)
        for e in (6.7, 6.9)
        for d in (9, 12)
    ]
    pmt += [{"hv_gain": 1e4, "dynode_stages": 9}]
    hyd = [(rate, tau) for rate in (1e3, 4e7, 1e9) for tau in (1.5, 0.5)]
    snr = [
        {"electrons": n, **params} for _, params in SNR_CASES for n in (0.5, 5, 50, 500)
    ]
    js = _run("detectors.mjs", {"emccd": emccd, "pmt": pmt, "hyd": hyd, "snr": snr})

    for (g, n), f in zip(emccd, js["emccd"], strict=True):
        expected = ms.CameraEMCCD(em_gain=g, em_stages=n).excess_noise_factor
        assert f == pytest.approx(expected, rel=1e-12)
    for p, (gain, f) in zip(pmt, js["pmt"], strict=True):
        tube = ms.PMT(**p, hv_range=None)
        assert gain == pytest.approx(tube.multiplication_gain, rel=1e-12)
        assert f == pytest.approx(tube.excess_noise_factor, rel=1e-12)
    for (rate, tau), eff in zip(hyd, js["hyd"], strict=True):
        expected = ms.HyD(dead_time_ns=tau)._counting_efficiency(np.array(rate), 1.0)
        assert eff == pytest.approx(float(expected), rel=1e-12)

    # the analytic SNR used in the docs, against microsim's detector simulation
    measured = [
        _measured_snr(det, n) for det, _ in SNR_CASES for n in (0.5, 5, 50, 500)
    ]
    np.testing.assert_allclose(js["snr"], measured, rtol=0.03)


CAMERA_CASES = [
    # (electrons per pixel, camera)
    (20, ms.CameraCCD(read_noise=6, electrons_per_adu=1, bit_depth=16)),
    (3, ms.CameraCMOS(read_noise=1.5, electrons_per_adu=0.5, bit_depth=16)),
    (200, ms.CameraCMOS(read_noise=0, electrons_per_adu=1, bit_depth=16)),
    (120, ms.CameraCCD(full_well=100, read_noise=2, electrons_per_adu=1)),
    (0.5, ms.CameraEMCCD(em_gain=300, read_noise=50, electrons_per_adu=20)),
    (5, ms.CameraEMCCD(em_gain=300, read_noise=50, electrons_per_adu=20)),
    (5, ms.CameraEMCCD(em_gain=2, read_noise=1, electrons_per_adu=1)),
]


def _camera_params(cam: ms.Detector) -> dict:
    return {
        "readNoise": cam.read_noise,
        "gain": cam.multiplication_gain,
        "enf": cam.excess_noise_factor,
        "electronsPerAdu": cam.conversion_factor,
        "offset": cam.offset,
        "bitDepth": cam.bit_depth,
        "fullWell": cam.full_well,
    }


def test_camera_simulation() -> None:
    size = 200_000
    payload = [
        {"electrons": n, "params": _camera_params(cam), "size": size}
        for n, cam in CAMERA_CASES
    ]
    js = _run("camera.mjs", payload)
    for (n, cam), result in zip(CAMERA_CASES, js, strict=True):
        cam = cam.model_copy(update={"dark_current": 0})
        xp = NumpyAPI()
        xp.set_random_seed(0)
        image = DataArray(np.full((1, 1, 1, size), n), dims=("c", "z", "y", "x"))
        py = np.asarray(cam.render(image, exposure_ms=1000, xp=xp), dtype=float)
        assert result["mean"] == pytest.approx(py.mean(), rel=0.01), cam
        assert result["var"] == pytest.approx(py.var(), rel=0.04), cam
        expected_q = np.quantile(py, [0.05, 0.5, 0.95])
        np.testing.assert_allclose(
            result["quantiles"], expected_q, atol=1.01, rtol=0.02
        )
