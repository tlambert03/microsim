// Detectors page, HyD section: fraction of photons recorded against the count rate
// (HyD._counting_efficiency, via physics/detectors.js).
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

const RATES = Array.from({ length: 121 }, (_, i) => 10 ** (5 + (i * 4) / 120)); // 1e5 to 1e9

export default async function (el) {
  const { hyd_dead_time_ns } = await loadData("detectors");
  const ui = shell(el);
  const state = { tau: hyd_dead_time_ns };
  slider(ui.controls, {
    label: "dead_time_ns",
    min: 0.25,
    max: 10,
    log: true,
    value: state.tau,
    format: (v) => `${fmt.sig(v, 2)} ns`,
    onInput: (v) => ((state.tau = v), chart.redraw()),
  });
  const setLoss = stat(ui.stats, { label: "counts lost at 40 Mcps" });
  const setMax = stat(ui.stats, { label: "maximum recorded rate", note: "1 / dead time" });

  const chart = responsive(ui.plot, (width) => {
    const rows = RATES.map((n) => ({ n, eff: D.hydCountingEfficiency(n, state.tau) }));
    setLoss(fmt.pct(1 - D.hydCountingEfficiency(4e7, state.tau)));
    setMax(`${fmt.si(D.hydMaxRate(state.tau))}<small>cps</small>`);
    const svg = Plot.plot({
      ...baseOptions(width),
      height: 220,
      marginLeft: 48,
      marginBottom: 38,
      x: { type: "log", label: "true photon rate (counts/s)", labelAnchor: "center", labelArrow: "none", labelOffset: 34, ticks: decades(1e5, 1e9), tickFormat: "~s" },
      y: { domain: [0, 1.02], label: "fraction of photons counted", labelArrow: "none", grid: true, tickFormat: ".0%" },
      marks: [
        yRangeMark(),
        Plot.ruleX([4e7], { stroke: C.faint, strokeDasharray: "2,3" }),
        Plot.lineY(rows, { x: "n", y: "eff", stroke: C.series[0], strokeWidth: 2.25 }),
      ],
    });
    ui.plot.replaceChildren(svg);
    crosshair(ui.plot, svg, {
      x: svg.scale("x"),
      content: (n) => {
        const eff = D.hydCountingEfficiency(n, state.tau);
        return tooltipHTML(`${fmt.si(n)} counts/s`, [
          { label: "recorded", value: fmt.si(n * eff), color: C.series[0] },
          { label: "lost", value: fmt.pct(1 - eff) },
        ]);
      },
    });
  });
}
