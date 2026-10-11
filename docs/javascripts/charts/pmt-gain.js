// Detectors page, PMT section: gain and excess noise factor of the PMT presets in
// detectors/lib.py against the control voltage (PMT.multiplication_gain and
// PMT.excess_noise_factor, via physics/detectors.js).
import * as D from "../physics/detectors.js";
import {
  C,
  Plot,
  baseOptions,
  crosshair,
  decades,
  fmt,
  loadData,
  responsive,
  select,
  shell,
  slider,
  stat,
  tooltipHTML,
  yRangeMark,
} from "./lib.js";

export default async function (el) {
  const { pmts } = await loadData("detectors");
  const ui = shell(el);
  const state = { pmt: pmts[0].name, hv: 0.8 };
  const redraw = () => chart.redraw();
  let hvSlider;
  select(ui.controls, {
    label: "Preset",
    options: pmts.map((p) => p.name),
    value: state.pmt,
    onInput: (v) => {
      state.pmt = v;
      const [lo, hi] = pmts.find((p) => p.name === v).hv_range;
      state.hv = Math.min(Math.max(state.hv, lo), hi);
      hvSlider.input.min = lo;
      hvSlider.set(state.hv);
      redraw();
    },
  });
  const [lo, hi] = pmts[0].hv_range;
  hvSlider = slider(ui.controls, {
    label: "hv (control voltage)",
    min: lo,
    max: hi,
    step: 0.01,
    value: state.hv,
    format: (v) => `${v.toFixed(2)} V`,
    onInput: (v) => ((state.hv = v), redraw()),
  });
  const setGain = stat(ui.stats, { label: "mean gain" });
  const setF = stat(ui.stats, { label: "excess noise factor F", note: "rises as the gain per dynode falls" });

  const panel = (width, rows, y, label, current, opts) => {
    const svg = Plot.plot({
      ...baseOptions(width),
      height: 210,
      marginLeft: 46,
      marginBottom: 38,
      x: { label: "control voltage (V)", labelAnchor: "center", labelArrow: "none", labelOffset: 34 },
      y: { label, labelArrow: "none", grid: true, ...opts },
      marks: [
        yRangeMark(),
        Plot.lineY(rows, { x: "hv", y, stroke: C.series[1], strokeWidth: 2.25 }),
        Plot.dot([current], { x: "hv", y, r: 5, fill: C.series[1], stroke: C.surface, strokeWidth: 2 }),
      ],
    });
    return svg;
  };

  const chart = responsive(ui.plot, (width) => {
    const p = pmts.find((x) => x.name === state.pmt);
    const at = (hv) => {
      const gain = D.pmtGain({ ...p, hv });
      return { hv, gain, f: D.pmtExcessNoiseFactor(gain, p.dynode_stages) };
    };
    const [a, b] = p.hv_range;
    const rows = Array.from({ length: 81 }, (_, i) => at(a + ((b - a) * i) / 80));
    const current = at(state.hv);
    setGain(fmt.si(current.gain));
    setF(fmt.sig(current.f, 3));

    const wide = width > 520;
    const w = wide ? Math.floor((width - 24) / 2) : width;
    const lo = 10 ** Math.floor(Math.log10(rows[0].gain));
    const hi = 10 ** Math.ceil(Math.log10(rows.at(-1).gain));
    const left = panel(w, rows, "gain", "mean gain", current, {
      type: "log",
      domain: [lo, hi],
      ticks: decades(lo, hi),
      tickFormat: "~s",
    });
    const fMax = Math.max(1.6, Math.ceil(Math.max(...rows.map((r) => r.f)) * 10) / 10);
    const right = panel(w, rows, "f", "excess noise factor F", current, { domain: [1, fMax] });
    const box = document.createElement("div");
    box.className = "ms-chart__panels";
    box.append(left, right);
    ui.plot.replaceChildren(box);
    for (const svg of [left, right]) {
      crosshair(ui.plot, svg, {
        x: svg.scale("x"),
        content: (hv) => {
          const r = at(hv);
          return tooltipHTML(`${hv.toFixed(2)} V`, [
            { label: "gain", value: fmt.si(r.gain), color: C.series[1] },
            { label: "F", value: fmt.sig(r.f, 3) },
          ]);
        },
      });
    }
  });
}
