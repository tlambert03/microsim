from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from microsim import schema as ms
from microsim._psf import (
    PSFGrid,
    airy_radius_um,
    apply_saturation,
    detection_psf,
    excitation_spot,
)
from microsim.psf import make_confocal_psf
from microsim.schema.dimensions import Axis
from microsim.schema.lens import ObjectiveLens
from microsim.schema.modality._optics import (
    FocusedIllumination,
    PinholeDetection,
    WidefieldIllumination,
)
from microsim.schema.modality._simple_psf import (
    bin_spectrum,
    emission_lines,
    excitation_lines,
)
from microsim.schema.optical_config import lib
from microsim.schema.optical_config.lib import (
    FITC,
    LASER_488,
    LASER_561,
    spectral_detector,
)
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
    [[fl]] = cf._channel_fluorescence()
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


def test_spectrum_product_on_different_grids() -> None:
    """Samples present in only one spectrum are interpolated, not zeroed."""
    wvl = np.arange(500, 600)
    fine = ms.Spectrum(wavelength=wvl, intensity=np.ones(wvl.size))
    for other in (wvl[::2], wvl + 0.5):  # coarser, or offset by half a sample
        half = ms.Spectrum(wavelength=other, intensity=np.full(other.size, 0.5))
        product = fine * half
        assert isinstance(product.intensity, np.ndarray)
        inside = np.arange(501, 598)
        np.testing.assert_allclose(product.at(inside), 0.5)


def test_detector_qe_without_emission_filter() -> None:
    oc = ms.OpticalConfig(lights=[ms.LightSource.laser(488)])
    full = oc.filtered_emission_rate(_green_fluor()).sum().item()
    half = oc.filtered_emission_rate(_green_fluor(), detector_qe=0.5).sum().item()
    assert half == pytest.approx(full / 2)


def test_bin_spectrum_keeps_all_samples() -> None:
    spectrum = xr.DataArray(
        [1.0, 2, 3, 4], dims=[Axis.W], coords={Axis.W: [500, 501, 502, 503]}
    )
    for bins in (1, 2, 3):
        assert float(bin_spectrum(spectrum, bins=bins).sum()) == 10


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
    fluorescence = sim._channel_fluorescence()
    [channel] = fluorescence
    detection = modality._detection(fluorescence, 0, sim.objective_lens)
    _key, psf, brightness = modality._system_psf(
        channel[0],
        detection,
        sim.ground_truth().attrs["space"],
        sim.objective_lens,
        sim.settings,
        sim._xp,
    )
    return psf, brightness


@pytest.mark.parametrize("power", [1e5, 1e7], ids=["linear", "saturated"])
def test_confocal_point_source_photometry(power: float) -> None:
    """An in-focus point source peaks at its detected rate x pinhole throughput."""
    wvl = np.arange(400, 650)
    narrow = ms.Fluorophore(  # emits at ~525 nm only, so one PSF wavelength is exact
        name="narrow",
        excitation_spectrum={
            "wavelength": wvl,
            "intensity": np.exp(-(((wvl - 490) / 20) ** 2)),
        },
        emission_spectrum={
            "wavelength": wvl,
            "intensity": np.exp(-(((wvl - 525) / 1.5) ** 2)),
        },
        extinction_coefficient=55_000,
        quantum_yield=0.6,
        lifetime_ns=2.6,
    )
    nz, nxy, dz, dx, n_fluors = 9, 65, 0.1, 0.03, 100
    truth = np.zeros((nz, nxy, nxy))
    truth[nz // 2, nxy // 2, nxy // 2] = n_fluors
    label = ms.FluorophoreDistribution.from_array(truth)
    sim = ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=truth.shape, scale=(dz, dx, dx)),
        sample=ms.Sample(labels=[label.model_copy(update={"fluorophore": narrow})]),
        channels=[
            ms.OpticalConfig(
                lights=[ms.LightSource.laser(488)],
                filters=[ms.Bandpass(bandcenter=525, bandwidth=50, placement="EM")],
                power=power,
            )
        ],
        objective_lens=ObjectiveLens(numerical_aperture=1.4),
        modality=ms.Confocal(pinhole_au=1),
        settings=ms.Settings(
            spectral_bins_per_emission_channel=1, float_dtype="float64", cache=False
        ),
    )
    img = np.asarray(sim.optical_image().data)
    # detected photons/s per fluorophore at the focus, including saturation there
    rate = float(sim.filtered_emission_rates().sum())
    psf = make_confocal_psf(
        nz=nz,
        nx=nxy,
        dz=dz,
        dxy=dx,
        ex_wvl_um=0.488,
        em_wvl_um=0.525,
        pinhole_au=1,
        objective=sim.objective_lens,
        normalize=False,
    )
    through_pinhole = float(psf.max())  # in focus, at the center of the spot
    assert 0.5 < through_pinhole < 1
    assert img.max() == pytest.approx(n_fluors * rate * through_pinhole, rel=1e-3)


@pytest.mark.parametrize("power", [1, 1e7], ids=["linear", "saturated"])
def test_confocal_psf_uses_excitation_wavelength(power: float) -> None:
    oc = ms.OpticalConfig(
        lights=[ms.LightSource.laser(488)], filters=FITC.filters, power=power
    )
    sim = _confocal_sim(oc, airy_unit_wavelength_nm=520)
    psf, brightness = _confocal_psf(sim)
    [[fl]] = sim._channel_fluorescence()
    assert brightness == pytest.approx(fl.detected_rate(saturate=False))

    # the excitation spot is computed at the laser wavelength, not the emission
    xp, objective = sim._xp, sim.objective_lens
    grid = PSFGrid(nz=9, nx=33, dz=0.1, dx=0.03)
    emission = emission_lines(fl.detected_fraction, bins=1, threshold_percentage=1)
    [(em_nm, _)] = emission
    pinhole_um = airy_radius_um(0.52, 1.4)
    expected = apply_saturation(
        excitation_spot(grid, 488, objective, xp), fl.saturation
    )
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


def test_emission_band_center_ignores_infrared_leaks() -> None:
    """Measured filter spectra often transmit again in the near-infrared."""
    wvl = np.arange(300, 1200)
    band = ((wvl >= 500) & (wvl <= 550)) * 0.95
    leak = ((wvl >= 930) & (wvl <= 1199)) * 0.9  # broad, and runs off the end
    spike = (wvl == 380) * 0.6
    em = ms.SpectrumFilter(
        transmission={"wavelength": wvl, "intensity": band + leak + spike},
        placement="EM",
    )
    oc = ms.OpticalConfig(filters=[em])
    assert oc.emission_band_center() == pytest.approx(525, abs=1)


@skipif_no_internet
def test_emission_band_center_fpbase() -> None:
    em = ms.SpectrumFilter.from_fpbase("Chroma ET525/50m", placement="EM")
    assert ms.OpticalConfig(filters=[em]).emission_band_center() == pytest.approx(
        525, abs=2
    )


def test_confocal_pinhole_matched_to_emission_band() -> None:
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
        fluorescence = sim._channel_fluorescence()
        det = cast("ms.Confocal", sim.modality)._detection(fluorescence, 0, objective)
        return cast("PinholeDetection", det)

    # one pinhole for both fluorophores, 2 AU at the band center
    expected = 2 * airy_radius_um(0.525, 1.4)
    assert _pinhole(oc).radius_um == pytest.approx(expected, rel=1e-3)
    custom = _pinhole(oc, airy_unit_wavelength_nm=600).radius_um
    assert custom == pytest.approx(2 * airy_radius_um(0.6, 1.4))

    # a longpass emission filter has no band center: fall back to detected light
    lp = ms.Longpass(cuton=500, placement="EM")
    open_ended = oc.model_copy(update={"filters": [lp]})
    with pytest.warns(UserWarning, match="no band-limited emission filter"):
        assert _pinhole(open_ended).radius_um > 0

    # a filter spectrum covering only its passband is band-limited
    passband = ms.SpectrumFilter(
        transmission={"wavelength": [500, 525, 550], "intensity": [1, 1, 1]},
        placement="EM",
    )
    narrow = oc.model_copy(update={"filters": [passband]})
    assert narrow.emission_band_center() == pytest.approx(525)
    assert _pinhole(narrow).radius_um == pytest.approx(expected, rel=1e-3)

    with pytest.raises(ValidationError):
        ms.Confocal(airy_unit_wavelength_nm=0)
    for closed in (0, 0.01):
        with pytest.raises(ValidationError):
            ms.Confocal(pinhole_au=closed)


def test_confocal_shared_pinhole() -> None:
    """Channels in the same track share one pinhole, sized for all their bands."""
    objective = ObjectiveLens(numerical_aperture=1.4)
    channels = spectral_detector(4, 500, 600, [488])  # 25 nm bins, 500-600 nm

    def _radii(tracks: Any = "simultaneous", **kwargs: Any) -> list[float]:
        sim = ms.Simulation(
            truth_space={"shape": (4, 4, 4), "scale": (1, 1, 1)},
            sample=[{"distribution": ms.MatsLines(), "fluorophore": FLAT}],
            channels=channels,
            objective_lens=objective,
            modality=ms.Confocal(**kwargs),
            tracks=tracks,
        )
        modality = cast("ms.Confocal", sim.modality)
        fluorescence = sim._channel_fluorescence()
        return [
            cast(
                "PinholeDetection", modality._detection(fluorescence, c, objective)
            ).radius_um
            for c in range(len(channels))
        ]

    per_channel = _radii(shared_pinhole=False)
    assert len(set(per_channel)) == len(channels)  # one pinhole per band
    assert _radii("sequential") == per_channel  # each channel is its own track
    shared = _radii()  # shared within the track by default
    assert len(set(shared)) == 1
    centers = [oc.emission_band_center() for oc in channels]
    expected = airy_radius_um(float(np.mean(centers)) / 1000, 1.4)  # type: ignore
    assert shared[0] == pytest.approx(expected, rel=1e-3)
    assert shared[0] == pytest.approx(airy_radius_um(0.55, 1.4), rel=0.01)

    # one pinhole per track
    two_tracks = _radii([[0, 1], [2, 3]])
    assert two_tracks[0] == two_tracks[1] < two_tracks[2] == two_tracks[3]


def test_confocal_channel_without_signal() -> None:
    """A channel that detects nothing is dark (no pinhole size is needed)."""
    far_red = ms.OpticalConfig(
        lights=[ms.LightSource.laser(700)],
        filters=[ms.Longpass(cuton=720, placement="EM")],
    )
    sim = ms.Simulation(
        truth_space={"shape": (4, 16, 16), "scale": (0.1, 0.05, 0.05)},
        sample=[{"distribution": ms.MatsLines(), "fluorophore": _green_fluor()}],
        channels=[FITC, far_red],
        modality=ms.Confocal(),
        settings=ms.Settings(cache=False),
    )
    img = sim.optical_image()
    assert float(img[0].sum()) > 0
    assert float(img[1].sum()) == 0


def test_single_wavelength_light() -> None:
    light = ms.LightSource(spectrum={"wavelength": [488], "intensity": [1]})
    sim = ms.Simulation(
        truth_space={"shape": (4, 16, 16), "scale": (0.1, 0.05, 0.05)},
        sample=[{"distribution": ms.MatsLines(), "fluorophore": _green_fluor()}],
        channels=[ms.OpticalConfig(lights=[light])],
        settings=ms.Settings(cache=False),
    )
    assert float(sim.optical_image().sum()) > 0


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


@pytest.mark.parametrize("laser", [405, 445, 488, 515, 561, 640])
def test_laser_channels(laser: int) -> None:
    ch = getattr(lib, f"LASER_{laser}")
    assert ch.emission_band_center() == pytest.approx(ch.filters[-1].bandcenter)
    # the dichroic passes the laser to the sample, and the emission path blocks it
    illum = ch.illumination
    assert illum.wavelength[np.argmax(illum.intensity)] == laser
    assert ch.emission.spectrum.at([laser])[0] < 1e-6


def _track_sim(
    channels: list[ms.OpticalConfig], tracks: Any, **kwargs: Any
) -> ms.Simulation:
    return ms.Simulation(
        truth_space={"shape": (4, 4, 4), "scale": (1, 1, 1)},
        sample=[{"distribution": ms.MatsLines(), "fluorophore": FLAT}],
        channels=channels,
        tracks=tracks,
        **kwargs,
    )


def test_tracks_by_index_or_name() -> None:
    channels = [LASER_488, LASER_561]
    assert _track_sim(channels, "sequential")._track_indices() == [[0], [1]]
    assert _track_sim(channels, "simultaneous")._track_indices() == [[0, 1]]
    assert _track_sim(channels, [["561"], [0]])._track_indices() == [[1], [0]]
    sim = _track_sim(channels, [[0, "561"]])
    assert sim.tracks == [ms.Track(channels=[0, "561"])]  # bare lists are cast
    assert ms.Simulation.model_validate_json(sim.model_dump_json()).tracks == sim.tracks

    bad = [
        ([[0]], "exactly one track"),  # channel 1 missing
        ([[0, 0], [1]], "exactly one track"),  # channel 0 twice
        ([["640"], [0, 1]], "matches no"),
        ([[2], [0, 1]], "out of range"),
    ]
    for tracks, match in bad:
        with pytest.raises(ValidationError, match=match):
            _track_sim(channels, tracks)
    with pytest.raises(ValidationError, match="matches several"):
        _track_sim([LASER_488, LASER_488], [["488"], [1]])


def test_simultaneous_track_excites_with_all_lasers() -> None:
    channels = [LASER_488, LASER_561]
    seq = _track_sim(channels, "sequential")._channel_fluorescence()
    sim = _track_sim(channels, "simultaneous")._channel_fluorescence()
    both = sum(row[0].total_absorption_rate(saturate=False) for row in seq)
    for c in range(2):
        assert sim[c][0].total_absorption_rate(saturate=False) == pytest.approx(both)
        # detection is still each channel's own
        assert sim[c][0].detected_fraction.equals(seq[c][0].detected_fraction)
    assert [row[0].track for row in seq] == [0, 1]
    assert [row[0].track for row in sim] == [0, 0]


def test_track_counts_a_shared_laser_once() -> None:
    # two detectors (emission windows) behind the same 488 nm laser
    em2 = ms.Bandpass(bandcenter=600, bandwidth=50, placement="EM")
    other_window = LASER_488.model_copy(
        update={"filters": [*LASER_488.filters[:-1], em2]}
    )
    channels = [LASER_488, other_window]
    seq = _track_sim(channels, "sequential")._channel_fluorescence()
    sim = _track_sim(channels, "simultaneous")._channel_fluorescence()
    for c in range(2):
        assert sim[c][0].total_absorption_rate(saturate=False) == pytest.approx(
            seq[c][0].total_absorption_rate(saturate=False)
        )
    # ... so the channels must agree on how the laser is set up
    brighter = other_window.model_copy(update={"power": 5})
    with pytest.raises(ValidationError, match="same power and excitation filters"):
        _track_sim([LASER_488, brighter], "simultaneous")
    _track_sim([LASER_488, brighter], "sequential")  # fine, if acquired separately
    two_lasers = LASER_488.model_copy(
        update={"lights": [*LASER_488.lights, *LASER_561.lights]}
    )
    with pytest.raises(ValidationError, match="some, but not all"):
        _track_sim([LASER_488, two_lasers], "simultaneous")


def test_track_shares_pixel_dwell() -> None:
    slow = LASER_561.model_copy(update={"exposure_ms": 0.01})
    with pytest.raises(ValidationError, match="same pixel dwell time"):
        _track_sim([LASER_488, slow], "simultaneous", modality=ms.Confocal())
    _track_sim([LASER_488, slow], "sequential", modality=ms.Confocal())
    # cameras can have different exposures
    _track_sim([LASER_488, slow], "simultaneous", modality=ms.Widefield())


def test_track_light_identity() -> None:
    # the same laser, written differently, is still one laser
    as_float = LASER_488.model_copy(update={"lights": [ms.LightSource.laser(488.0)]})
    em2 = ms.Bandpass(bandcenter=600, bandwidth=50, placement="EM")
    as_float = as_float.model_copy(update={"filters": [*LASER_488.filters[:-1], em2]})
    seq = _track_sim([LASER_488, as_float], "sequential")._channel_fluorescence()
    sim = _track_sim([LASER_488, as_float], "simultaneous")._channel_fluorescence()
    assert sim[0][0].total_absorption_rate(saturate=False) == pytest.approx(
        seq[0][0].total_absorption_rate(saturate=False)
    )
    # ... and the order of a channel's lights doesn't matter
    lasers = [*LASER_488.lights, *LASER_561.lights]
    a = LASER_488.model_copy(update={"lights": lasers})
    b = as_float.model_copy(update={"lights": lasers[::-1]})
    _track_sim([a, b], "simultaneous")


def test_track_lamp_channels() -> None:
    # channels without lights (a lamp through excitation filters) behave the same
    em2 = ms.Bandpass(bandcenter=600, bandwidth=50, placement="EM")
    other_window = FITC.model_copy(update={"filters": [*FITC.filters[:-1], em2]})
    assert other_window.excitation == FITC.excitation  # same excitation side
    seq = _track_sim([FITC, other_window], "sequential")._channel_fluorescence()
    sim = _track_sim([FITC, other_window], "simultaneous")._channel_fluorescence()
    assert sim[0][0].total_absorption_rate(saturate=False) == pytest.approx(
        seq[0][0].total_absorption_rate(saturate=False)
    )
    brighter = other_window.model_copy(update={"power": 5})
    with pytest.raises(ValidationError, match="same power and excitation filters"):
        _track_sim([FITC, brighter], "simultaneous")


def test_simultaneous_confocal_adds_up_lasers() -> None:
    """Without saturation, a track's image is the sum of what each laser excites."""
    sim = ms.Simulation(
        truth_space={"shape": (8, 32, 32), "scale": (0.1, 0.05, 0.05)},
        sample=[{"distribution": ms.MatsLines(), "fluorophore": FLAT}],  # no lifetime
        channels=[LASER_488, LASER_561],
        tracks="simultaneous",
        modality=ms.Confocal(),
        settings=ms.Settings(float_dtype="float64", cache=False),
    )
    both = sim.optical_image_per_fluor()
    by_laser = [
        ms.ChannelFluorescence.from_channel(oc, FLAT).absorption_rate
        for oc in sim.channels
    ]

    def excited_by(absorption: xr.DataArray) -> xr.DataArray:
        # every channel's own detection, excited by `absorption` alone
        fluorescence = [
            [
                ms.ChannelFluorescence.create(
                    oc, FLAT, sim.objective_lens, absorption_rate=absorption, track=0
                )
            ]
            for oc in sim.channels
        ]
        args = (sim.objective_lens, sim.settings, ms.NumpyAPI())
        return sim.modality.render(sim.ground_truth(), fluorescence, *args)

    total = sum(excited_by(a) for a in by_laser)
    np.testing.assert_allclose(both, total, rtol=1e-5)
    assert not np.allclose(both, excited_by(by_laser[0]))


def test_channel_fluorescence_without_track_has_own_pinhole() -> None:
    channels = spectral_detector(2, 500, 600, [488])
    objective = ObjectiveLens(numerical_aperture=1.4)
    fluorescence = [
        [ms.ChannelFluorescence.create(oc, FLAT, objective)] for oc in channels
    ]
    modality = ms.Confocal()
    radii = [
        cast("PinholeDetection", modality._detection(fluorescence, c, objective))
        for c in range(2)
    ]
    assert radii[0].radius_um < radii[1].radius_um


def test_rejected_assignment_is_undone() -> None:
    sim = _track_sim([LASER_488, LASER_561], [[0], [1]])
    with pytest.raises(ValidationError, match="out of range"):
        sim.channels = [LASER_488]
    assert sim.channels == [LASER_488, LASER_561]
    with pytest.raises(ValidationError, match="exactly one track"):
        sim.tracks = [[0]]
    assert sim.tracks == [ms.Track(channels=[0]), ms.Track(channels=[1])]
    sim = ms.Simulation(
        truth_space={"shape": (4, 4, 4), "scale": (1, 1, 1)},
        sample=[ms.MatsLines()],
        channels=[LASER_488, LASER_561],
    )
    with pytest.raises(ValidationError):
        sim.tracks = [[0]]
    assert "tracks" not in sim.model_fields_set


def test_track_validation() -> None:
    channels = [LASER_488, LASER_561]
    for bad in ([[0, 1], []], [[True], [0]]):
        with pytest.raises(ValidationError):
            _track_sim(channels, bad)
