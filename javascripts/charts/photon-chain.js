// Photon budget page: the photon budget as two products of spectra, one plot each.
// Excitation: light at the sample × absorption cross section = absorbed.
// Emission: emitted × emission path × QE × collection = detected.
// Factors are drawn relative to their peaks (transmission and QE as they are), so
// each shaded product is the product of the curves drawn with it.  The
// calculation is microsim's (docs/javascripts/physics/photometry.js); this file
// only chooses the parameters and draws.
import * as P from "../physics/photometry.js";
import {
  C,
  Plot,
  baseOptions,
  crosshair,
  d3,
  fmt,
  legendItem,
  responsive,
  select,
  shell,
  slider,
  tooltipHTML,
  yRangeMark,
} from "./lib.js";
import { loadPhotometry } from "./photometry-data.js";

const toRows = (s) => s.w.map((w, i) => ({ w, v: s.v[i] }));
const count = (x) => (x >= 1e5 ? fmt.si(x) : x >= 100 ? fmt.int(x) : fmt.sig(x, 3));
const peak = (s) => Math.max(...s.v) || 1;
const relative = (s) => P.scale(s, 1 / peak(s));
const constant = (w, v) => ({ w, v: w.map(() => v) });

function valueAt(s, nm) {
  const i = d3.bisector((d) => d).center(s.w, nm);
  return Math.abs(s.w[i] - nm) <= 1 ? s.v[i] : 0;
}

// Wavelength range where any of `spectra` exceeds `frac` of its own peak, padded.
function extent(spectra, frac, pad = 15) {
  let [lo, hi] = [Infinity, -Infinity];
  for (const s of spectra) {
    const p = peak(s);
    s.w.forEach((w, i) => {
      if (s.v[i] > frac * p) [lo, hi] = [Math.min(lo, w), Math.max(hi, w)];
    });
  }
  return [Math.max(300, lo - pad), Math.min(800, hi + pad)];
}

// One plot: `lines` are `{spectrum, color, dash, label}`, `product` the shaded
// `{spectrum, color, label}`; `tip(nm)` gives the tooltip rows.
function panel(parent, width, { title, total, domain, lines, product, tip }) {
  const box = document.createElement("div");
  box.className = "ms-chain__col";
  box.innerHTML = `<div class="ms-chain__label"><span class="ms-chain__title">${title}</span>${total}</div>`;
  const legend = document.createElement("div");
  legend.className = "ms-chart__legend";
  box.append(legend);
  for (const l of lines) legendItem(legend, { label: l.label, color: l.color, swatch: l.dash ? "dash" : "line" });
  legendItem(legend, { label: product.label, color: product.color, swatch: "area" });
  parent.append(box);

  const svg = Plot.plot({
    ...baseOptions(width),
    height: Math.max(200, Math.min(260, width * 0.6)),
    marginLeft: 40,
    marginRight: 8,
    marginBottom: 36,
    x: { domain, label: "wavelength (nm)", labelAnchor: "center", labelArrow: "none", labelOffset: 30 },
    y: { domain: [0, 1.05], label: null, grid: true, ticks: 5, tickFormat: d3.format(".0%") },
    marks: [
      yRangeMark(),
      Plot.areaY(toRows(product.spectrum), { x: "w", y: "v", fill: product.color, fillOpacity: 0.35, clip: true }),
      Plot.lineY(toRows(product.spectrum), { x: "w", y: "v", stroke: product.color, strokeWidth: 2, clip: true }),
      ...lines.map((l) =>
        Plot.lineY(toRows(l.spectrum), {
          x: "w",
          y: "v",
          stroke: l.color,
          strokeWidth: 1.5,
          strokeDasharray: l.dash,
          clip: true,
        }),
      ),
      Plot.ruleY([0], { stroke: C.faint }),
    ],
  });
  const plot = document.createElement("div");
  plot.className = "ms-chart__plot";
  plot.append(svg);
  box.append(plot);
  crosshair(plot, svg, { x: svg.scale("x"), content: (nm) => tooltipHTML(`${Math.round(nm)} nm`, tip(nm)) });
}

export default async function (el) {
  const { fluors, channels, detectors, lens } = await loadPhotometry();
  const ui = shell(el);
  const state = { fluor: "EGFP", channel: "FITC (Chroma 49002)", detector: detectors[0].name, power: 1 };
  const redraw = () => chart.redraw();
  const pick = (label, key, options) =>
    select(ui.controls, { label, options, value: state[key], onInput: (v) => ((state[key] = v), redraw()) });
  pick("Fluorophore", "fluor", fluors.map((f) => f.name));
  pick("Channel", "channel", channels.map((c) => c.name));
  pick("Detector QE", "detector", detectors.map((d) => d.name));
  slider(ui.controls, {
    label: "Irradiance",
    min: 0.01,
    max: 1e6,
    log: true,
    value: state.power,
    format: (v) => `${fmt.sig(v, 2)} W/cm²`,
    onInput: (v) => ((state.power = v), redraw()),
  });

  const chart = responsive(ui.plot, (width) => {
    const fluor = fluors.find((f) => f.name === state.fluor);
    const channel = channels.find((c) => c.name === state.channel);
    const det = detectors.find((d) => d.name === state.detector);
    const qy = fluor.qy || 1;

    const irrad = P.irradiance(channel.illumination, state.power);
    const flux = P.fluxDensity(irrad);
    const sigma = P.crossSection(fluor.ex, fluor.ec);
    const eps = P.extinction(fluor.ex, fluor.ec);
    const b = P.photonBudget({
      channel,
      fluor,
      power: state.power,
      qe: det.qe,
      na: lens.na,
      specimenRI: lens.specimen_ri,
    });
    const k = P.total(b.absorbed);
    const nEm = P.total(b.emitted);
    const nDet = P.total(b.collected);
    const path = channel.emission ?? constant(b.emitted.w, 1);
    const qe = typeof det.qe === "number" ? constant(path.w, det.qe) : det.qe;
    const sat = b.saturation > 0.005 ? ` × ${fmt.sig(nEm / (qy * k), 2)} saturation` : "";

    // scaled by the factors' peaks, so each product is the product of the curves
    const absorbedRel = P.scale(b.absorbed, 1 / (peak(flux) * peak(sigma)));
    const detectedRel = P.scale(b.collected, 1 / peak(b.emitted));

    const cols = width > 560 ? 2 : 1;
    const colWidth = cols === 2 ? Math.floor((width - 24) / 2) : width;
    const grid = document.createElement("div");
    grid.className = "ms-chain";
    grid.style.setProperty("--cols", cols);
    ui.plot.replaceChildren(grid);

    panel(grid, colWidth, {
      title: "Excitation",
      total: `<span>absorbed <b>${count(k)}/s</b></span>`,
      domain: extent([flux, sigma], 0.01),
      lines: [
        { spectrum: relative(flux), color: C.ink, label: "light at the sample" },
        { spectrum: relative(sigma), color: C.muted, dash: "4,3", label: "absorption cross section" },
      ],
      product: { spectrum: absorbedRel, color: C.series[0], label: "absorbed (product)" },
      tip: (nm) => [
        { label: "light", value: `${fmt.sci(valueAt(irrad, nm))} W/cm²`, color: C.ink },
        { label: "σ", value: `${fmt.sci(valueAt(sigma, nm))} cm²`, color: C.muted },
        { label: "ε", value: `${fmt.int(valueAt(eps, nm))} M⁻¹cm⁻¹`, color: C.muted },
        { label: "absorbed", value: `${fmt.sci(valueAt(b.absorbed, nm))} /s`, color: C.series[0] },
      ],
    });
    panel(grid, colWidth, {
      title: "Emission",
      total: `<span>emitted <b>${count(nEm)}/s</b> → detected <b>${count(nDet)}/s</b></span>`,
      domain: extent([b.emitted, path], 0.02),
      lines: [
        { spectrum: relative(b.emitted), color: C.series[1], label: `emitted (absorbed × QY ${qy}${sat})` },
        { spectrum: path, color: C.ink, label: "emission path" },
        { spectrum: qe, color: C.muted, dash: "4,3", label: "detector QE" },
      ],
      product: {
        spectrum: detectedRel,
        color: C.series[2],
        label: `detected (product, × ${fmt.pct(b.collection)} collected)`,
      },
      tip: (nm) => [
        { label: "emitted", value: `${fmt.sci(valueAt(b.emitted, nm))} /s`, color: C.series[1] },
        { label: "emission path", value: fmt.pct(valueAt(path, nm)), color: C.ink },
        { label: "QE", value: fmt.pct(valueAt(qe, nm)), color: C.muted },
        { label: "detected", value: `${fmt.sci(valueAt(b.collected, nm))} /s`, color: C.series[2] },
      ],
    });
  });
}
