// Detectors page, EMCCD section: excess noise factor against EM gain
// (CameraEMCCD.excess_noise_factor, via physics/detectors.js), starting at the
// EMCCD of detector_data in scripts/make_doc_figures.py.
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
  shell,
  slider,
  stat,
  tooltipHTML,
  yRangeMark,
} from "./lib.js";

const GAINS = Array.from({ length: 151 }, (_, i) => 10 ** ((i * 3) / 150)); // 1 to 1000

export default async function (el) {
  const { cameras } = await loadData("detectors");
  const ui = shell(el);
  const state = { stages: cameras.EMCCD.em_stages, gain: cameras.EMCCD.em_gain };
  const redraw = () => chart.redraw();
  slider(ui.controls, {
    label: "EM gain",
    min: 1,
    max: 1000,
    log: true,
    value: state.gain,
    format: (v) => fmt.int(v),
    onInput: (v) => ((state.gain = v), redraw()),
  });
  slider(ui.controls, {
    label: "em_stages",
    min: 1,
    max: 1000,
    log: true,
    value: state.stages,
    format: (v) => fmt.int(Math.round(v)),
    onInput: (v) => ((state.stages = Math.round(v)), redraw()),
  });
  const setF = stat(ui.stats, { label: "excess noise factor F" });
  const setQE = stat(ui.stats, { label: "equivalent QE", note: "1 / F: the QE of a noise-free detector with the same SNR" });

  const chart = responsive(ui.plot, (width) => {
    const rows = GAINS.map((g) => ({ g, f: D.emccdExcessNoiseFactor(g, state.stages) }));
    const f = D.emccdExcessNoiseFactor(state.gain, state.stages);
    setF(fmt.sig(f, 3));
    setQE(`${fmt.pct(1 / f)}<small> of the QE</small>`);
    const svg = Plot.plot({
      ...baseOptions(width),
      height: 220,
      marginLeft: 44,
      marginBottom: 38,
      x: { type: "log", label: "EM gain", labelAnchor: "center", labelArrow: "none", labelOffset: 34, ticks: decades(1, 1000) },
      y: { domain: [0.9, 2.1], label: "F", labelArrow: "none", grid: true },
      marks: [
        yRangeMark(),
        Plot.ruleY([2], { stroke: C.faint, strokeDasharray: "3,3" }),
        Plot.lineY(rows, { x: "g", y: "f", stroke: C.series[1], strokeWidth: 2.25 }),
        Plot.dot([{ g: state.gain, f }], { x: "g", y: "f", r: 5, fill: C.series[1], stroke: C.surface, strokeWidth: 2 }),
      ],
    });
    ui.plot.replaceChildren(svg);
    crosshair(ui.plot, svg, {
      x: svg.scale("x"),
      content: (g) =>
        tooltipHTML(`gain ${fmt.sig(g, 3)}`, [
          { label: "F", value: fmt.sig(D.emccdExcessNoiseFactor(g, state.stages), 3), color: C.series[1] },
        ]),
    });
  });
}
