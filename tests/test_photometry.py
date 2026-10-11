from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError

from microsim import schema as ms
from microsim.psf import (
    PSFGrid,
    airy_radius_um,
    detection_psf,
    excitation_spot,
    make_confocal_psf,
    saturate,
)
from microsim.schema.lens import ObjectiveLens
from microsim.schema.modality._optics import (
    FocusedIllumination,
    PinholeDetection,
    WidefieldIllumination,
)
from microsim.schema.modality._simple_psf import emission_lines, excitation_lines
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


def test_emission_rates_saturation() -> None:
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

    # rates are for a fluorophore at the reference intensity, for any modality
    rates = cf.filtered_emission_rates()
    np.testing.assert_allclose(wf.filtered_emission_rates(), rates)
    unsaturated = cf.filtered_emission_rates(saturate=False)
    np.testing.assert_allclose(rates, unsaturated / (1 + s), atol=1e-300)
    [[fl]] = cf.channel_fluorescence()
    assert fl.saturation == pytest.approx(s)
    assert fl.detected_rate(saturate=False) == pytest.approx(float(unsaturated.sum()))
    assert fl.detected_rate() == pytest.approx(float(rates.sum()))


def test_off_grid_laser_wavelength() -> None:
    """Lasers between the sampled wavelengths of filters/spectra still excite."""
    fluor = _green_fluor()
    rates = [
        ms.OpticalConfig(lights=[ms.LightSource.laser(w)], filters=FITC.filters)
        .absorption_rate(fluor)
        .sum()
        .item()
        for w in (488, 488.5)
    ]
    assert rates[0] > 0
    assert rates[1] == pytest.approx(rates[0], rel=1e-3)


def _confocal_sim(oc: ms.OpticalConfig, **kwargs: Any) -> ms.Simulation:
    return ms.Simulation(
        truth_space={"shape": (9, 33, 33), "scale": (0.1, 0.03, 0.03)},
        sample=[{"distribution": ms.MatsLines(), "fluorophore": _green_fluor()}],
        channels=[oc],
        objective_lens=ObjectiveLens(numerical_aperture=1.4),
        modality=ms.Confocal(**kwargs),
        settings=ms.Settings(max_psf_radius_aus=None, cache=False),
    )


def _confocal_psf(sim: ms.Simulation) -> tuple[Any, float]:
    modality = cast("ms.Confocal", sim.modality)
    [channel] = sim.channel_fluorescence()
    detection = modality._detection(channel, sim.objective_lens)
    _key, psf, brightness = modality._system_psf(
        channel[0],
        detection,
        sim.ground_truth().attrs["space"],
        sim.objective_lens,
        sim.settings,
        sim._xp,
    )
    return psf, brightness


@pytest.mark.parametrize("power", [1, 1e7], ids=["linear", "saturated"])
def test_confocal_psf_uses_excitation_wavelength(power: float) -> None:
    oc = ms.OpticalConfig(
        lights=[ms.LightSource.laser(488)], filters=FITC.filters, power=power
    )
    sim = _confocal_sim(oc, airy_unit_wavelength_nm=520)
    psf, brightness = _confocal_psf(sim)
    [[fl]] = sim.channel_fluorescence()
    assert brightness == pytest.approx(fl.detected_rate(saturate=False))

    # the excitation spot is computed at the laser wavelength, not the emission
    xp, objective = sim._xp, sim.objective_lens
    grid = PSFGrid(nz=9, nx=33, dz=0.1, dx=0.03)
    emission = emission_lines(fl.detection, bins=1, threshold_percentage=1)
    [(em_nm, _)] = emission
    pinhole_um = airy_radius_um(0.52, 1.4)
    expected = saturate(excitation_spot(grid, 488, objective, xp), fl.saturation)
    expected = expected * detection_psf(grid, em_nm, objective, xp, pinhole_um)
    np.testing.assert_allclose(psf, expected, rtol=1e-3)
    assert psf.max() == pytest.approx(
        detection_psf(grid, em_nm, objective, xp, pinhole_um).max()
        / (1 + fl.saturation),
        rel=1e-3,
    )


def test_confocal_polychromatic_excitation() -> None:
    """Excitation spots are weighted by absorption, summed, then saturated."""
    fluor = _green_fluor()
    lasers = [ms.LightSource.laser(w) for w in (405, 488, 491)]
    oc = ms.OpticalConfig(lights=lasers)
    lines = excitation_lines(oc.absorption_rate(fluor).squeeze())
    # 405 is excited too weakly to pass the threshold; 488 and 491 share a bin
    assert len(lines) == 1
    assert 488 < lines[0][0] < 491
    assert lines[0][1] == 1

    # broadband excitation is split into bins of at most `bin_nm`
    broad = excitation_lines(FITC.absorption_rate(fluor).squeeze(), bin_nm=10)
    assert len(broad) > 1
    assert np.all(np.diff([w for w, _ in broad]) < 2 * 10)
    assert sum(f for _, f in broad) == pytest.approx(1, abs=1e-3)

    xp, objective = ms.NumpyAPI(), ObjectiveLens(numerical_aperture=1.4)
    grid = PSFGrid(nz=9, nx=33, dz=0.1, dx=0.03)
    blue = excitation_spot(grid, 405, objective, xp)
    green = excitation_spot(grid, 488, objective, xp)
    assert blue[4].sum() < green[4].sum()  # shorter wavelength: tighter spot
    mixed = FocusedIllumination().spot(((405, 1), (488, 3)), grid, objective, xp)
    np.testing.assert_allclose(mixed, 0.25 * blue + 0.75 * green, rtol=1e-6)
    assert mixed.max() == pytest.approx(1)
    assert WidefieldIllumination().spot(((405, 1),), grid, objective, xp) == 1


def test_confocal_pinhole_matched_to_emission_band(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """1 AU is defined at the center of the emission filter band, per channel."""
    objective = ObjectiveLens(numerical_aperture=1.4)
    green, flat = _green_fluor(), FLAT
    bp = ms.Bandpass(bandcenter=525, bandwidth=50, placement="EM")
    oc = ms.OpticalConfig(lights=[ms.LightSource.laser(488)], filters=[bp])

    def _pinhole(oc: ms.OpticalConfig, **kwargs: Any) -> PinholeDetection:
        sim = ms.Simulation(
            truth_space={"shape": (4, 4, 4), "scale": (1, 1, 1)},
            sample=[
                {"distribution": ms.MatsLines(), "fluorophore": green},
                {"distribution": ms.MatsLines(), "fluorophore": flat},
            ],
            channels=[oc],
            objective_lens=objective,
            modality=ms.Confocal(pinhole_au=2, **kwargs),
        )
        [channel] = sim.channel_fluorescence()
        det = cast("ms.Confocal", sim.modality)._detection(channel, objective)
        return cast("PinholeDetection", det)

    # one pinhole for both fluorophores, 2 AU at the band center
    expected = 2 * airy_radius_um(0.525, 1.4)
    assert _pinhole(oc).radius_um == pytest.approx(expected, rel=1e-3)
    custom = _pinhole(oc, airy_unit_wavelength_nm=600).radius_um
    assert custom == pytest.approx(2 * airy_radius_um(0.6, 1.4))

    # a longpass emission filter has no band center: fall back to detected light
    lp = ms.Longpass(cuton=500, placement="EM")
    open_ended = oc.model_copy(update={"filters": [lp]})
    with caplog.at_level("WARNING"):
        assert _pinhole(open_ended).radius_um > 0
    assert "no band-limited emission filter" in caplog.text


def test_emission_binning_preserves_brightness() -> None:
    """Splitting emission into more PSF wavelengths doesn't change total photons."""

    def _total(bins: int) -> float:
        sim = ms.Simulation(
            truth_space={"shape": (8, 64, 64), "scale": (0.1, 0.05, 0.05)},
            sample=[{"distribution": ms.MatsLines(), "fluorophore": _green_fluor()}],
            settings=ms.Settings(
                spectral_bins_per_emission_channel=bins, random_seed=0, cache=False
            ),
        )
        return float(sim.optical_image().sum())

    assert _total(4) == pytest.approx(_total(1), rel=1e-3)


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
