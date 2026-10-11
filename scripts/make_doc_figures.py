"""Generate the figures used in the documentation.

Every figure is produced by microsim itself, so the docs show what the code does.
Images go to `docs/images/generated/`, including the figures of the tutorials,
which are made by running each tutorial page's code.  Data for the interactive
charts (`docs/javascripts/charts/`) goes to `docs/data/generated/`.

Usage (some figures fetch data from FPbase and OpenOrganelle, so need network):

    uv run --group docs python scripts/make_doc_figures.py [figure ...]
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import warnings
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from microsim import schema as ms
from microsim._data_array import DataArray
from microsim.psf import make_psf
from microsim.schema.backend import NumpyAPI
from microsim.schema.detectors import lib as det_lib
from microsim.schema.optical_config import lib as oc_lib

if TYPE_CHECKING:
    from collections.abc import Callable

mpl.use("Agg")
warnings.filterwarnings("ignore")

DOCS = Path(__file__).parents[1] / "docs"
OUT = DOCS / "images" / "generated"
DATA = DOCS / "data" / "generated"
FIGURES: dict[str, Callable[[], None]] = {}

# image panels are dark in both schemes, so their labels use a neutral gray
IMG_LABEL = "#9a998f"


def figure(func: Callable[[], None]) -> Callable[[], None]:
    FIGURES[func.__name__] = func
    return func


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("wrote", path.relative_to(OUT.parents[2]))


def write_json(name: str, obj: object) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{name}.json"
    path.write_text(json.dumps(obj, separators=(",", ":")))
    print("wrote", path.relative_to(DOCS.parent), f"({path.stat().st_size // 1024} kB)")


def spectrum_json(spec: ms.Spectrum) -> dict:
    """Compact spectrum for docs/javascripts/physics (decoded by `decodeSpectrum`).

    Values are rounded to 5 significant digits, and below 1e-9 to 0.  Uniformly
    spaced wavelengths are stored as a start `w0` and step `dw`.
    """
    w = np.asarray(spec.wavelength, dtype=float)
    v = [0 if abs(x) < 1e-9 else float(f"{x:.5g}") for x in np.asarray(spec.intensity)]
    steps = np.diff(w)
    if len(w) > 1 and np.all(steps == steps[0]):
        return {"w0": float(w[0]), "dw": float(steps[0]), "v": v}
    return {"w": w.tolist(), "v": v}


def image_json(img: np.ndarray) -> dict:
    """An image normalized to its maximum, as base64 little-endian uint16."""
    q = np.round(np.clip(img / img.max(), 0, 1) * 65535).astype("<u2")
    return {"shape": list(q.shape), "data": base64.b64encode(q.tobytes()).decode()}


def norm(img: np.ndarray, lo: float = 0.1, hi: float = 99.9) -> np.ndarray:
    a, b = np.percentile(img, (lo, hi))
    return np.clip((img - a) / (b - a + 1e-12), 0, 1)


def composite(channels: list[np.ndarray], colors: list[str]) -> np.ndarray:
    """Additively combine normalized grayscale channels into an RGB image."""
    rgb = np.zeros((*channels[0].shape, 3))
    for img, color in zip(channels, colors, strict=False):
        rgb += norm(img)[..., None] * np.array(mpl.colors.to_rgb(color))
    return np.clip(rgb, 0, 1)


def image_axes(ax: plt.Axes, title: str = "") -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, color=IMG_LABEL, fontsize=10, loc="left")


def _np(x: object) -> np.ndarray:
    data = getattr(x, "data", x)
    if hasattr(data, "get"):
        data = data.get()
    return np.asarray(data)


# ---------------------------------------------------------------------------
# Figures


def _cell_sim() -> ms.Simulation:
    return ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=(64, 512, 512), scale=0.064),
        output_space={"downscale": 2},
        sample=[
            ms.FluorophoreDistribution(
                distribution=ms.CosemLabel(dataset="jrc_hela-3", label="er-mem_pred"),
                fluorophore="EGFP",
            ),
            ms.FluorophoreDistribution(
                distribution=ms.CosemLabel(dataset="jrc_hela-3", label="mito-mem_pred"),
                fluorophore="mCherry",
            ),
        ],
        channels=[oc_lib.FITC, oc_lib.TRITC],
        detector=det_lib.ICX285,
        exposure_ms=5,
        settings=ms.Settings(random_seed=0),
    )


@figure
def pipeline() -> None:
    """Ground truth -> optical image -> digital image, for one z-plane."""
    sim = _cell_sim()
    z = 18
    truth = _np(sim.ground_truth())[:, z]
    optical = sim.optical_image()
    digital = _np(sim.digital_image(optical))[:, z // 2]
    optical = _np(optical)[:, z]

    colors = ["#33ff66", "#ff33cc"]
    panels = [
        ("1  ground truth  (fluorophores / voxel)", composite(list(truth), colors)),
        ("2  optical image  (photons / s)", composite(list(optical), colors)),
        ("3  digital image  (gray values)", composite(list(digital), colors)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, (title, img) in zip(axes, panels, strict=True):
        ax.imshow(img, interpolation="nearest")
        image_axes(ax, title)
    fig.subplots_adjust(wspace=0.03)
    save(fig, OUT / "pipeline.png")

    # zoomed crop, showing pixelation in the output space
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.3))
    crop = np.s_[200:280, 330:410]
    crop2 = np.s_[100:140, 165:205]
    axes[0].imshow(panels[1][1][crop], interpolation="nearest")
    image_axes(axes[0], "optical image, truth space (64 nm voxels)")
    axes[1].imshow(panels[2][1][crop2], interpolation="nearest")
    image_axes(axes[1], "digital image, output space (128 nm pixels)")
    fig.subplots_adjust(wspace=0.03)
    save(fig, OUT / "spaces-crop.png")


# offered in the interactive spectra and photon budget charts, blue to far red
CHART_FLUOROPHORES = [
    "DAPI",
    "mTagBFP2",
    "mTurquoise2",
    "EGFP",
    "mNeonGreen",
    "Alexa Fluor 488",
    "mVenus",
    "tdTomato",
    "mRFP1",
    "mScarlet-I",
    "Alexa Fluor 568",
    "mCherry",
    "Alexa Fluor 647",
]


# FITC with a longpass emission filter: collects more green, and more crosstalk
FITC_LP = ms.OpticalConfig(
    name="FITC, 500 LP emission",
    filters=[
        ms.Bandpass(bandcenter=470, bandwidth=40, placement="EX"),
        ms.Longpass(cuton=495, placement="BS"),
        ms.Longpass(cuton=500, placement="EM"),
    ],
)


def _filter_label(f: ms.Filter) -> str:
    if f.name:
        return f.name
    if isinstance(f, ms.Bandpass):
        return f"{f.bandcenter:g}/{f.bandwidth:g} bandpass"
    if isinstance(f, ms.Longpass):
        return f"{f.cuton:g} longpass"
    return type(f).__name__


@figure
def photometry_data() -> None:
    """Spectra and photophysics for the interactive spectra and photon budget charts."""
    fluors = []
    for name in CHART_FLUOROPHORES:
        f = ms.Fluorophore.from_fpbase(name)
        fluors.append(
            {
                "name": name,
                "ec": f.extinction_coefficient,
                "qy": f.quantum_yield,
                "lifetime_ns": f.lifetime_ns,
                "ex": spectrum_json(f.excitation_spectrum),
                "em": spectrum_json(f.emission_spectrum),
            }
        )
    configs = [
        *(getattr(oc_lib, n) for n in ["DAPI", "ECFP", "FITC", "EYFP", "TRITC"]),
        FITC_LP,
        *(getattr(oc_lib, n) for n in ["DSRED", "CY5", "CY7"]),
        *(
            ms.OpticalConfig.from_fpbase(microscope_id="wKqWbg", config_name=n)
            for n in ["Widefield Green", "Widefield Red"]
        ),
    ]
    channels = []
    for oc in configs:
        channels.append(
            {
                "name": oc.name,
                "filters": [
                    {
                        "name": _filter_label(f),
                        "placement": f.placement.value,
                        "spectrum": spectrum_json(f.spectrum),
                    }
                    for f in oc.filters
                ],
                "lights": [
                    {
                        "name": li.name,
                        "power": li.power,
                        "spectrum": spectrum_json(li.spectrum),
                    }
                    for li in oc.lights
                ],
                "illumination": spectrum_json(oc.illumination),
                "emission": spectrum_json(
                    oc.emission.spectrum if oc.emission else None
                ),
            }
        )
    detectors = [
        {"name": "QE = 1 (no detector)", "qe": 1},
        {"name": "ICX285 CCD", "qe": spectrum_json(det_lib.ICX285.qe)},
        {"name": "GaAsP PMT", "qe": spectrum_json(det_lib.PMT_GAASP.qe)},
        {"name": "Power HyD R", "qe": spectrum_json(det_lib.POWER_HYD_R.qe)},
    ]
    write_json(
        "photometry",
        {"fluorophores": fluors, "channels": channels, "detectors": detectors},
    )


# cameras in the interactive SNR chart, with the parameters it starts from
SNR_CAMERAS = {
    "CCD": ms.CameraCCD(read_noise=6),
    "sCMOS": ms.CameraCMOS(read_noise=1.5),
    "EMCCD": ms.CameraEMCCD(em_gain=300, read_noise=50),
}


def _measured_snr(detector: ms.Detector, electrons: float) -> float:
    """SNR of a flat image recorded by `detector` (unclipped), in a 1 s exposure."""
    exact = {"bit_depth": 32, "electrons_per_adu": 0.01, "offset": 10**6}
    det = detector.model_copy(update={**exact, "dark_current": 0, "full_well": 10**9})
    xp = NumpyAPI()
    xp.set_random_seed(0)
    image = DataArray(np.full((1, 1, 2000, 2000), electrons), dims=("c", "z", "y", "x"))
    gray = np.asarray(det.render(image, exposure_ms=1000, xp=xp), dtype=float)
    signal = (gray - det.offset) * det.conversion_factor
    return float(signal.mean() / signal.std())


@figure
def detector_data() -> None:
    """Detector presets, and simulated SNR, for the interactive detector charts."""
    levels = np.logspace(-1, 4, 11)
    simulated = {
        name: [
            {"electrons": float(n), "snr": _measured_snr(det, float(n))} for n in levels
        ]
        for name, det in SNR_CAMERAS.items()
    }
    cameras = {
        name: {
            "read_noise": det.read_noise,
            "em_gain": getattr(det, "em_gain", 1),
            "em_stages": getattr(det, "em_stages", None),
        }
        for name, det in SNR_CAMERAS.items()
    }
    pmts = [
        {
            "name": pmt.name,
            "ref_hv": pmt.ref_hv,
            "ref_gain": pmt.ref_gain,
            "hv_exponent": pmt.hv_exponent,
            "dynode_stages": pmt.dynode_stages,
            "hv_range": pmt.hv_range,
        }
        for pmt in (det_lib.PMT_GAASP, det_lib.PMT_MULTIALKALI)
    ]
    write_json(
        "detectors",
        {
            "cameras": cameras,
            "simulated_snr": simulated,
            "pmts": pmts,
            "hyd_dead_time_ns": det_lib.HYD_SP8.dead_time_ns,
        },
    )


def _lines_sim(**kwargs: object) -> ms.Simulation:
    return ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=(64, 256, 256), scale=(0.04, 0.02, 0.02)),
        output_space={"downscale": 4},
        sample=[ms.MatsLines(density=0.4, length=30, azimuth=5, max_r=1)],
        settings=ms.Settings(random_seed=1, max_psf_radius_aus=6),
        **kwargs,
    )


@figure
def modalities() -> None:
    """The same sample through widefield and confocal (XY and XZ)."""
    sims = {
        "widefield": _lines_sim(),
        "confocal, 1 AU": _lines_sim(modality=ms.Confocal(pinhole_au=1)),
    }
    fig, axes = plt.subplots(
        2, 2, figsize=(8, 5.8), gridspec_kw={"height_ratios": [4, 1.3], "hspace": 0.1}
    )
    for col, (name, sim) in enumerate(sims.items()):
        img = _np(sim.optical_image())[0]
        mid = img.shape[0] // 2
        axes[0, col].imshow(norm(img[mid]), cmap="gray")
        image_axes(axes[0, col], f"{name}  ·  XY")
        xz = img[:, img.shape[1] // 2]
        axes[1, col].imshow(norm(xz), cmap="gray", aspect=2)
        image_axes(axes[1, col], "XZ")
    save(fig, OUT / "modalities.png")


@figure
def noise_data() -> None:
    """Noise-free expected image (ER channel of the HeLa cell), for the noise chart."""
    sim = _cell_sim()
    expected = sim.digital_image(sim.optical_image(), with_detector_noise=False)
    write_json("noise", {"image": image_json(_np(expected)[0, 9].astype(float))})


PINHOLES_AU = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3]


def _fwhm(profile: np.ndarray, step: float) -> float:
    """Full width at half maximum of a peaked 1D profile, linearly interpolated."""
    half = profile.max() / 2
    i = int(np.argmax(profile))
    lo = i
    while lo > 0 and profile[lo - 1] > half:
        lo -= 1
    hi = i
    while hi < len(profile) - 1 and profile[hi + 1] > half:
        hi += 1
    left = lo - (profile[lo] - half) / (profile[lo] - profile[lo - 1])
    right = hi + (profile[hi] - half) / (profile[hi] - profile[hi + 1])
    return float((right - left) * step)


@figure
def pinhole_data() -> None:
    """Widefield and confocal PSFs across pinhole sizes, for the pinhole chart."""
    obj = ms.ObjectiveLens(numerical_aperture=1.4)
    # fine enough to resolve small pinholes, and wide enough (6 µm) that defocused
    # widefield planes (up to ±0.96 µm) keep their light
    dx, dz = 0.02, 0.04
    kw = {"nz": 49, "nx": 301, "dx": dx, "dz": dz, "objective": obj, "em_wvl_nm": 525}
    psfs = {"widefield": make_psf(**kw)}
    for au in PINHOLES_AU:
        psfs[au] = make_psf(ex_wvl_nm=488, pinhole_au=au, **kw)

    entries = []
    for key, psf in psfs.items():
        p = _np(psf).astype(float)
        cz, cy = p.shape[0] // 2, p.shape[1] // 2
        crop = slice(cy - 40, cy + 41, 2)  # 1.6 µm, displayed at 40 nm
        planes = p.sum(axis=(1, 2))
        entries.append(
            {
                "pinhole_au": None if key == "widefield" else key,
                "peak": float(p[cz, cy, cy]),
                "fwhm_lateral_nm": 1000 * _fwhm(p[cz, cy, :], dx),
                "fwhm_axial_nm": 1000 * _fwhm(p[:, cy, cy], dz),
                "plane_sums": [round(float(x), 5) for x in planes / planes.max()],
                "xy": image_json(p[cz, crop, crop]),
                "xz": image_json(p[cz - 20 : cz + 21, cy, crop]),  # 1.6 µm square
            }
        )
    write_json(
        "pinhole",
        {
            "na": obj.numerical_aperture,
            "ex_nm": 488,
            "em_nm": 525,
            "image_dx_um": 2 * dx,
            "dz_um": dz,
            "psfs": entries,
        },
    )


@figure
def crosstalk() -> None:
    """EGFP (ER) and tdTomato (mito) in a bandpass and a longpass green channel."""
    sim = ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=(64, 512, 512), scale=0.064),
        sample=[
            ms.FluorophoreDistribution(
                distribution=ms.CosemLabel(dataset="jrc_hela-3", label="er-mem_pred"),
                fluorophore="EGFP",
            ),
            ms.FluorophoreDistribution(
                distribution=ms.CosemLabel(dataset="jrc_hela-3", label="mito-mem_pred"),
                fluorophore="tdTomato",
            ),
        ],
        channels=[oc_lib.FITC, FITC_LP],
        settings=ms.Settings(random_seed=0),
    )
    per_fluor = _np(sim.optical_image_per_fluor())  # (c, f, z, y, x)
    z = 18
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.7), gridspec_kw={"hspace": 0.12})
    for row, oc in enumerate(sim.channels):
        totals = per_fluor[row].sum(axis=(1, 2, 3))
        share = totals / totals.sum()
        panels = [
            (per_fluor[row, 0, z], f"from EGFP (ER): {share[0]:.1%}"),
            (per_fluor[row, 1, z], f"from tdTomato (mito): {share[1]:.1%}"),
            (per_fluor[row, :, z].sum(axis=0), f"{oc.name}: recorded"),
        ]
        # one display range per row, so each panel shows its true share
        vmax = np.percentile(panels[2][0], 99.8)
        for ax, (img, title) in zip(axes[row], panels, strict=True):
            ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
            image_axes(ax, title)
    save(fig, OUT / "crosstalk.png")


def _save_plain(fig: plt.Figure, name: str) -> None:
    """Save a figure made by microsim's own plotting functions (white background)."""
    path = OUT / name
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.relative_to(DOCS.parent))


@figure
def spectra_plots() -> None:
    """The output of the plotting methods on the channels page."""
    oc = ms.OpticalConfig.from_fpbase("wKqWbg", "Widefield Green")
    oc.plot(show=False)
    _save_plain(plt.gcf(), "plot-config.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
    oc.plot_excitation(ax=axes[0])
    oc.plot_emission(ax=axes[1])
    _save_plain(fig, "plot-paths.png")

    sim = ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=(8, 64, 64), scale=0.05),
        sample=[
            ms.FluorophoreDistribution(distribution=ms.MatsLines(), fluorophore=f)
            for f in ("EGFP", "mCherry")
        ],
        channels=[oc_lib.FITC, oc_lib.TRITC],
    )
    sim.plot()
    _save_plain(plt.gcf(), "plot-simulation.png")


@figure
def first_simulation() -> None:
    """The example from the home page and the "First simulation" page."""
    sim = ms.Simulation(
        truth_space=ms.ShapeScaleSpace(shape=(128, 512, 512), scale=(0.02, 0.01, 0.01)),
        output_space={"downscale": 8},
        sample=ms.Sample(labels=[ms.MatsLines(density=0.5, length=30, azimuth=5)]),
        detector=ms.CameraCMOS(),
        settings=ms.Settings(random_seed=42),
    )
    truth = sim.ground_truth()
    optical = sim.optical_image()
    digital = sim.digital_image(optical)
    print(digital.shape, float(digital.min()), float(digital.max()))
    panels = [
        ("sim.ground_truth()  ·  max projection", _np(truth)[0].max(0)),
        ("sim.optical_image()  ·  z = 64", _np(optical)[0, 64]),
        ("sim.digital_image()  ·  z = 8", _np(digital)[0, 8]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), gridspec_kw={"wspace": 0.04})
    for ax, (title, img) in zip(axes, panels, strict=True):
        ax.imshow(norm(img), cmap="gray", interpolation="nearest")
        image_axes(ax, title)
    save(fig, OUT / "first-simulation.png")

    # brighter illumination, then confocal
    imgs = [("widefield, 1 W/cm², 100 ms  (defaults)", _np(digital)[0, 8])]
    sim.channels = [oc_lib.FITC.model_copy(update={"power": 10})]
    imgs.append(
        ("widefield, 10 W/cm², 100 ms", _np(sim.digital_image(optical * 10))[0, 8])
    )
    sim.modality = ms.Confocal(pinhole_au=0.5)
    sim.exposure_ms = 0.01
    sim.channels = [oc_lib.FITC.model_copy(update={"power": 50_000})]
    img = sim.digital_image()
    imgs.append(("confocal, 50 kW/cm², 10 µs dwell", _np(img)[0, 8]))
    for title, im in imgs:
        print(title, im.max())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), gridspec_kw={"wspace": 0.04})
    for ax, (title, img) in zip(axes, imgs, strict=True):
        ax.imshow(norm(img), cmap="gray", interpolation="nearest")
        image_axes(ax, title)
    save(fig, OUT / "first-simulation-brighter.png")


@figure
def tutorials() -> None:
    """Run the code blocks of each tutorial page, and save each figure made.

    The page `tutorials/<name>.md` shows them as
    `images/generated/tutorials/<name>/01.png`, `02.png`, and so on.
    """
    # the pages don't all set a seed; fix it here so builds are reproducible
    os.environ.setdefault("MICROSIM_RANDOM_SEED", "0")
    for page in sorted((DOCS / "tutorials").glob("*.md")):
        text = page.read_text()
        expected = len(re.findall(rf"generated/tutorials/{page.stem}/\d+\.png", text))
        if not expected:
            continue
        out = OUT / "tutorials" / page.stem
        out.mkdir(parents=True, exist_ok=True)
        plt.close("all")
        namespace: dict = {}
        n = 0
        for block in re.findall(r"```python([^`]*)```", text, re.DOTALL):
            exec(dedent(block), namespace)
            for num in plt.get_fignums():
                n += 1
                fig = plt.figure(num)
                path = out / f"{n:02d}.png"
                fig.savefig(path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                print("wrote", path.relative_to(DOCS.parent))
        if n != expected:
            raise RuntimeError(f"{page.name} made {n} figures, but shows {expected}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(FIGURES)
    for name in names:
        print(f"--- {name}")
        FIGURES[name]()
