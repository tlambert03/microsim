// Photon budget page: absorbed, emitted, and detected photons for one fluorophore
// in one channel, and a running tally down to photons per exposure.
// The calculation is microsim's (docs/javascripts/physics/photometry.js); this file
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

function valueAt(s, nm) {
  const i = d3.bisector((d) => d).center(s.w, nm);
  return Math.abs(s.w[i] - nm) <= 1 ? s.v[i] : 0;
}

// A row of tiles joined by arrows.  Returns one setter per tile.
function flow(parent, labels) {
  const row = document.createElement("div");
  row.className = "ms-flow";
  parent.append(row);
  return labels.map((label) => {
    const tile = document.createElement("div");
    tile.className = "ms-flow__step";
    tile.innerHTML = `<div class="ms-flow__value"></div><div class="ms-flow__label">${label}</div><div class="ms-flow__factor"></div>`;
    row.append(tile);
    const [value, , factor] = tile.children;
    return (v, f) => {
      value.innerHTML = v;
      factor.innerHTML = f;
    };
  });
}

export default async function (el) {
  const { fluors, channels, detectors, lens } = await loadPhotometry();
  const ui = shell(el);
  const state = {
    fluor: "EGFP",
    channel: "FITC",
    detector: detectors[0].name,
    power: 1, // W/cm^2
    na: lens.na, // ObjectiveLens defaults
    ns: lens.specimen_ri,
    exposure: 100, // ms
  };
  const redraw = () => chart.redraw();

  const row1 = document.createElement("div");
  row1.className = "ms-chart__controls";
  ui.controls.replaceWith(row1);
  const row2 = document.createElement("div");
  row2.className = "ms-chart__controls ms-chart__controls--grid";
  row1.after(row2);

  select(row1, {
    label: "Fluorophore",
    options: fluors.map((f) => f.name),
    value: state.fluor,
    onInput: (v) => ((state.fluor = v), redraw()),
  });
  select(row1, {
    label: "Channel",
    options: channels.map((c) => c.name),
    value: state.channel,
    onInput: (v) => ((state.channel = v), redraw()),
  });
  select(row1, {
    label: "Detector QE",
    options: detectors.map((d) => d.name),
    value: state.detector,
    onInput: (v) => ((state.detector = v), redraw()),
  });
  slider(row2, {
    label: "Irradiance",
    min: 0.01,
    max: 1e6,
    log: true,
    value: state.power,
    format: (v) => `${fmt.sig(v, 2)} W/cm²`,
    onInput: (v) => ((state.power = v), redraw()),
  });
  slider(row2, {
    label: "NA",
    min: 0.1,
    max: 1.49,
    step: 0.01,
    value: state.na,
    format: (v) => v.toFixed(2),
    onInput: (v) => ((state.na = v), redraw()),
  });
  slider(row2, {
    label: "Exposure",
    min: 0.001,
    max: 1000,
    log: true,
    value: state.exposure,
    format: (v) => `${fmt.sig(v, 2)} ms`,
    onInput: (v) => ((state.exposure = v), redraw()),
  });

  const colors = { absorbed: C.series[0], emitted: C.series[1], detected: C.series[2] };
  legendItem(ui.legend, { label: "absorbed", color: colors.absorbed });
  legendItem(ui.legend, { label: "emitted", color: colors.emitted });
  legendItem(ui.legend, {
    label: "detected (collected, filtered, × QE)",
    color: colors.detected,
    swatch: "area",
  });

  const [setAbs, setEm, setColl, setFilt, setDet, setExp] = flow(ui.stats, [
    "absorbed",
    "emitted",
    "collected",
    "pass the filters",
    "detected",
    "per exposure",
  ]);

  const chart = responsive(ui.plot, (width) => {
    const fluor = fluors.find((f) => f.name === state.fluor);
    const channel = channels.find((c) => c.name === state.channel);
    const qe = detectors.find((d) => d.name === state.detector).qe;

    // Simulation.filtered_emission_rates for one fluorophore and channel, with and
    // without the detector QE
    const opts = { channel, fluor, power: state.power, na: state.na, specimenRI: state.ns };
    const b = P.photonBudget({ ...opts, qe });
    const noQE = P.photonBudget({ ...opts, qe: null });
    const { absorbed, emitted, collection: eta, saturation: s } = b;
    const detected = b.collected;

    const k = P.total(absorbed);
    const nEm = P.total(emitted);
    const nColl = nEm * eta;
    const nPass = P.total(noQE.collected);
    const nDet = P.total(b.collected);
    const nExp = (nDet * state.exposure) / 1000;
    const qy = fluor.qy || 1;
    const satNote =
      s > 0.005
        ? `<br><span class="ms-flow__warn">× ${fmt.sig(nEm / (qy * k), 2)} saturation (kτ = ${fmt.sig(s, 2)})</span>`
        : !fluor.lifetime_ns
          ? "<br>lifetime unknown: no saturation"
          : "";
    setAbs(`${count(k)}<small>/s</small>`, `σ × photon flux`);
    setEm(`${count(nEm)}<small>/s</small>`, `× QY ${qy}${satNote}`);
    setColl(`${count(nColl)}<small>/s</small>`, `× ${fmt.pct(eta)} into the objective`);
    setFilt(`${count(nPass)}<small>/s</small>`, `× ${fmt.pct(nPass / nColl || 0)} emission path`);
    setDet(
      `${count(nDet)}<small>/s</small>`,
      typeof qe === "number" && qe === 1 ? "× QE 1" : `× ${fmt.pct(nDet / nPass || 0)} mean QE`,
    );
    setExp(`${count(nExp)}`, `× ${fmt.sig(state.exposure, 2)} ms, per molecule`);

    const series = [
      ...toRows(absorbed).map((d) => ({ ...d, key: "absorbed" })),
      ...toRows(emitted).map((d) => ({ ...d, key: "emitted" })),
    ];
    const lo = Math.min(absorbed.w.find((_, i) => absorbed.v[i] > 0) ?? 350, 450);
    const svg = Plot.plot({
      ...baseOptions(width),
      height: Math.max(220, Math.min(300, width * 0.38)),
      marginLeft: 52,
      marginBottom: 38,
      x: {
        domain: [Math.max(lo - 20, 300), 800],
        label: "wavelength (nm)",
        labelAnchor: "center",
        labelArrow: "none",
        labelOffset: 34,
      },
      y: { label: "photons / s / nm", labelArrow: "none", grid: true, ticks: 5, tickFormat: "~s" },
      color: { domain: ["absorbed", "emitted"], range: [colors.absorbed, colors.emitted] },
      marks: [
        yRangeMark(),
        Plot.areaY(toRows(detected), { x: "w", y: "v", fill: colors.detected, fillOpacity: 0.35, clip: true }),
        Plot.lineY(toRows(detected), { x: "w", y: "v", stroke: colors.detected, strokeWidth: 2, clip: true }),
        Plot.lineY(series, { x: "w", y: "v", z: "key", stroke: "key", strokeWidth: 2, clip: true }),
        Plot.ruleY([0], { stroke: C.faint }),
      ],
    });
    ui.plot.replaceChildren(svg);
    crosshair(ui.plot, svg, {
      x: svg.scale("x"),
      content: (nm) =>
        tooltipHTML(`${Math.round(nm)} nm`, [
          { label: "absorbed", value: fmt.sig(valueAt(absorbed, nm), 3), color: colors.absorbed },
          { label: "emitted", value: fmt.sig(valueAt(emitted, nm), 3), color: colors.emitted },
          { label: "detected", value: fmt.sig(valueAt(detected, nm), 3), color: colors.detected },
        ]),
    });
  });
}
