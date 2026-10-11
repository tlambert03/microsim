"""Generate the figures used in the documentation.

Every figure is produced by microsim itself, so the docs show what the code does.
Output goes to `docs/images/generated/`.  Line plots are written twice, with
`-light` and `-dark` suffixes, for the two color schemes of the docs.

Usage (some figures fetch data from FPbase and OpenOrganelle, so need network):

    uv run --group docs python scripts/make_doc_figures.py [figure ...]
"""

from __future__ import annotations

# ruff: noqa: RUF001
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
from microsim.psf import make_psf
from microsim.schema.detectors import lib as det_lib
from microsim.schema.optical_config import lib as oc_lib

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

mpl.use("Agg")
warnings.filterwarnings("ignore")

DOCS = Path(__file__).parents[1] / "docs"
OUT = DOCS / "images" / "generated"
FIGURES: dict[str, Callable[[], None]] = {}

# categorical palette (fixed order), and text inks, for light and dark schemes
THEMES = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "fill": "#8a8984",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#3a3a38",
        "fill": "#9a998f",
    },
}
# image panels are dark in both schemes, so their labels use a neutral gray
IMG_LABEL = "#9a998f"


def figure(func: Callable[[], None]) -> Callable[[], None]:
    FIGURES[func.__name__] = func
    return func


def themed(name: str) -> Iterator[dict]:
    """Yield a palette per color scheme, with matching rcParams active."""
    for scheme, t in THEMES.items():
        rc = {
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": t["text"],
            "axes.labelcolor": t["muted"],
            "axes.edgecolor": t["grid"],
            "axes.facecolor": "none",
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
            "xtick.color": t["muted"],
            "ytick.color": t["muted"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": t["grid"],
            "grid.linewidth": 0.6,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "legend.labelcolor": t["text"],
        }
        with mpl.rc_context(rc):
            yield {**t, "path": OUT / f"{name}-{scheme}.png"}


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("wrote", path.relative_to(OUT.parents[2]))


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


@figure
def spectra() -> None:
    """EGFP spectra and the FITC filter set."""
    egfp = ms.Fluorophore.from_fpbase("EGFP")
    fitc = oc_lib.FITC
    ex = egfp.excitation_spectrum
    em = egfp.emission_spectrum
    for t in themed("spectra-egfp-fitc"):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(8, 4.6), sharex=True, gridspec_kw={"hspace": 0.15}
        )
        s = t["series"]
        ax1.plot(ex.wavelength, ex.intensity / ex.intensity.max(), color=s[0])
        ax1.plot(em.wavelength, em.intensity / em.intensity.max(), color=s[2])
        ax1.text(445, 0.82, "EGFP excitation", color=t["text"], ha="right")
        ax1.text(545, 0.82, "EGFP emission", color=t["text"], ha="left")
        ax1.set_ylabel("relative intensity")
        ax1.set_ylim(0, 1.08)

        styles = {"EX": (s[0], "-"), "BS": (t["fill"], "--"), "EM": (s[2], "-")}
        for f in fitc.filters:
            color, ls = styles[f.placement.value]
            sp = f.spectrum
            ax2.plot(sp.wavelength, sp.intensity, color=color, linestyle=ls, lw=1.5)
            if ls == "-":
                ax2.fill_between(sp.wavelength, sp.intensity, color=color, alpha=0.15)
        ax2.text(
            470,
            0.5,
            "excitation\nfilter",
            color=t["text"],
            ha="center",
            va="center",
            fontsize=9,
        )
        ax2.text(
            525,
            0.5,
            "emission\nfilter",
            color=t["text"],
            ha="center",
            va="center",
            fontsize=9,
        )
        ax2.text(
            560,
            0.72,
            "dichroic: reflects excitation,\ntransmits emission",
            color=t["muted"],
            ha="left",
            va="top",
            fontsize=9,
        )
        ax2.set_ylabel("transmission")
        ax2.set_ylim(0, 1.15)
        ax2.set_xlim(380, 720)
        ax2.set_xlabel("wavelength (nm)")
        save(fig, t["path"])


@figure
def photon_budget() -> None:
    """Absorption, emission, and filtered emission rates for EGFP in FITC."""
    egfp = ms.Fluorophore.from_fpbase("EGFP")
    fitc = oc_lib.FITC
    obj = ms.ObjectiveLens(numerical_aperture=1.4)
    absr = fitc.absorption_rate(egfp).squeeze()
    emr = fitc.total_emission_rate(egfp).squeeze()
    filt = fitc.filtered_emission_rate(egfp).squeeze()
    coll = filt * obj.collection_efficiency
    for t in themed("photon-budget"):
        fig, ax = plt.subplots(figsize=(8, 3.4))
        s = t["series"]
        rows = [
            (absr, s[0], "absorbed"),
            (emr, s[1], "emitted"),
            (filt, s[2], "passes emission filters"),
            (coll, s[4], "collected by a 1.4 NA objective"),
        ]
        for da, color, label in rows:
            w = da.coords["w"].values
            ax.plot(w, da.values, color=color, label=f"{label}: {float(da.sum()):.0f}")
        ax.set_xlim(400, 650)
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("photons / s / nm")
        ax.legend(
            title="per EGFP molecule, total photons/s",
            alignment="left",
            fontsize=9,
            title_fontsize=9,
            loc="upper right",
        )
        save(fig, t["path"])


@figure
def psf() -> None:
    """Widefield vs confocal PSFs (XZ and XY sections)."""
    obj = ms.ObjectiveLens(numerical_aperture=1.4)
    kw = {"nz": 65, "nx": 65, "dx": 0.04, "dz": 0.04, "objective": obj}
    psfs = {
        "widefield": make_psf(em_wvl_nm=525, **kw),
        "confocal, 1 AU": make_psf(ex_wvl_nm=488, em_wvl_nm=525, pinhole_au=1, **kw),
        "confocal, 0.25 AU": make_psf(
            ex_wvl_nm=488, em_wvl_nm=525, pinhole_au=0.25, **kw
        ),
    }
    fig, axes = plt.subplots(
        2, 3, figsize=(9, 6.2), gridspec_kw={"hspace": 0.12, "wspace": 0.05}
    )
    for col, (name, p) in enumerate(psfs.items()):
        p = _np(p)
        p = p / p.max()
        mid = p.shape[0] // 2
        kws = {"cmap": "magma", "vmin": 0, "vmax": 1, "interpolation": "bilinear"}
        axes[0, col].imshow(p[mid] ** 0.5, **kws)
        image_axes(axes[0, col], f"{name}  ·  XY")
        axes[1, col].imshow(p[:, mid] ** 0.5, **kws)
        image_axes(axes[1, col], "XZ")
    # 1 µm scale bar
    axes[1, 0].plot([5, 30], [60, 60], color="white", lw=2)
    axes[1, 0].text(17.5, 57, "1 µm", color="white", ha="center", fontsize=8)
    save(fig, OUT / "psf-comparison.png")


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
def detectors() -> None:
    """One dim optical image through different cameras."""
    sim = _lines_sim()
    optical = sim.optical_image()
    # scale so the brightest output pixel expects ~20 photons in the exposure
    expected = sim.digital_image(optical, with_detector_noise=False)
    optical = optical * (20 / float(expected.max()))
    cams = {
        "expected photons (no noise)": None,
        "CCD, 6 e⁻ read noise": ms.CameraCCD(
            qe=1, read_noise=6, electrons_per_adu=1, bit_depth=16
        ),
        "sCMOS, 1.5 e⁻ read noise": ms.CameraCMOS(
            qe=1, read_noise=1.5, electrons_per_adu=0.5, bit_depth=16
        ),
        "EMCCD, EM gain 300": ms.CameraEMCCD(
            qe=1, read_noise=50, em_gain=300, electrons_per_adu=20, bit_depth=16
        ),
    }
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), gridspec_kw={"wspace": 0.04})
    for ax, (name, det) in zip(axes, cams.items(), strict=True):
        sim.detector = det
        img = _np(sim.digital_image(optical, with_detector_noise=det is not None))
        ax.imshow(norm(img[0, img.shape[1] // 2], 0.5, 99.8), cmap="gray")
        image_axes(ax, name)
    save(fig, OUT / "detectors.png")


@figure
def detector_curves() -> None:
    """EMCCD excess noise factor vs gain, and PMT gain vs HV."""
    gains = np.logspace(0, 3, 200)
    enf = [ms.CameraEMCCD(em_gain=g).excess_noise_factor for g in gains]
    hv = np.linspace(0.5, 0.9, 100)
    pmt = det_lib.PMT_GAASP
    pmt_gain = [pmt.model_copy(update={"hv": v}).multiplication_gain for v in hv]
    pmt_enf = [pmt.model_copy(update={"hv": v}).excess_noise_factor for v in hv]
    for t in themed("detector-curves"):
        s = t["series"]
        fig, (a1, a2, a3) = plt.subplots(
            1, 3, figsize=(11, 3.2), gridspec_kw={"wspace": 0.35}
        )
        a1.semilogx(gains, enf, color=s[0])
        a1.set_xlabel("EM gain")
        a1.set_ylabel("excess noise factor F")
        a1.set_title("EMCCD", loc="left", color=t["text"])
        a1.set_ylim(0.9, 2.1)
        a2.semilogy(hv, pmt_gain, color=s[1])
        a2.set_xlabel("control voltage (V)")
        a2.set_ylabel("mean gain")
        a2.set_title("GaAsP PMT: gain", loc="left", color=t["text"])
        a3.plot(hv, pmt_enf, color=s[1])
        a3.set_xlabel("control voltage (V)")
        a3.set_ylabel("excess noise factor F")
        a3.set_title("GaAsP PMT: noise", loc="left", color=t["text"])
        a3.set_ylim(0.9, 2.1)
        save(fig, t["path"])


@figure
def point_detectors() -> None:
    """PMT vs HyD on a confocal simulation (as in examples/point_detectors.py)."""
    sim = ms.Simulation(
        truth_space={"upscale": 4},
        output_space=ms.ShapeScaleSpace(shape=(1, 128, 128), scale=(0.16, 0.08, 0.08)),
        sample=[
            ms.FluorophoreDistribution(
                distribution=ms.MatsLines(density=0.5, length=30, azimuth=5, max_r=1),
                fluorophore="EGFP",
                concentration=1,
            )
        ],
        channels=[oc_lib.FITC.model_copy(update={"power": 2e4})],
        modality=ms.Confocal(pinhole_au=1),
        settings=ms.Settings(random_seed=100, max_psf_radius_aus=8),
        detector=det_lib.HYD_SP8,
    )
    optical = sim.optical_image()
    dets = {
        "GaAsP PMT, 0.7 V": det_lib.PMT_GAASP.model_copy(update={"hv": 0.7}),
        "GaAsP PMT, 0.9 V": det_lib.PMT_GAASP.model_copy(update={"hv": 0.9}),
        "HyD (counting)": det_lib.HYD_SP8,
        "HyD, 4× averaging": det_lib.HYD_SP8.model_copy(update={"averaging": 4}),
    }
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), gridspec_kw={"wspace": 0.04})
    for ax, (name, det) in zip(axes, dets.items(), strict=True):
        sim.detector = det
        img = _np(sim.digital_image(optical, exposure_ms=0.001))
        ax.imshow(img[0, 0], cmap="gray")
        image_axes(ax, f"{name}\nmax gray value: {int(img.max())}")
    save(fig, OUT / "point-detectors.png")


@figure
def own_truth() -> None:
    """A user-provided ground truth array, simulated in widefield and confocal."""
    shape = (64, 256, 256)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    rng = np.random.default_rng(0)
    truth = np.zeros(shape, dtype=np.float32)
    for _ in range(12):
        cz, cy, cx = rng.integers(16, 48), rng.integers(30, 226), rng.integers(30, 226)
        r = rng.integers(8, 28)
        d = np.sqrt(((z - cz) * 2) ** 2 + (y - cy) ** 2 + (x - cx) ** 2)
        truth += (np.abs(d - r) < 1.5) * 10  # hollow shells, 10 fluors / voxel

    common = {
        "scale": (0.04, 0.02, 0.02),
        "output_space": {"downscale": 4},
        "detector": ms.CameraCMOS(qe=0.8, read_noise=1.5),
        "settings": ms.Settings(random_seed=0),
    }
    wf = ms.Simulation.from_ground_truth(truth, **common).run()
    cf_img = ms.Simulation.from_ground_truth(
        truth,
        modality=ms.Confocal(),
        channels=[oc_lib.FITC.model_copy(update={"power": 5000})],
        exposure_ms=0.01,
        **common,
    ).run()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), gridspec_kw={"wspace": 0.04})
    axes[0].imshow(norm(truth[32]), cmap="gray")
    image_axes(axes[0], "your array (z = 32)")
    axes[1].imshow(norm(_np(wf)[0, 8]), cmap="gray")
    image_axes(axes[1], "widefield")
    axes[2].imshow(norm(_np(cf_img)[0, 8]), cmap="gray")
    image_axes(axes[2], "confocal")
    save(fig, OUT / "own-truth.png")


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
def tutorial() -> None:
    """Run the code blocks of the two-color tutorial, and save each figure made."""
    page = DOCS / "tutorials" / "two-color-cell.md"
    text = page.read_text()
    blocks = re.findall(r"```python([^`]*)```", text, re.DOTALL)
    out = OUT / "tutorial"
    out.mkdir(exist_ok=True)
    # the page doesn't set a seed; fix it here so builds are reproducible
    os.environ.setdefault("MICROSIM_RANDOM_SEED", "0")
    namespace: dict = {}
    n = 0
    for block in blocks:
        exec(dedent(block), namespace)
        for num in plt.get_fignums():
            n += 1
            fig = plt.figure(num)
            fig.savefig(out / f"tut-{n:02d}.png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            print("wrote", (out / f"tut-{n:02d}.png").relative_to(DOCS.parent))
    expected = len(re.findall(r"generated/tutorial/tut-\d+\.png", text))
    if n != expected:
        raise RuntimeError(f"tutorial made {n} figures, but the page shows {expected}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(FIGURES)
    for name in names:
        print(f"--- {name}")
        FIGURES[name]()
