// Detectors page: SNR of a CCD, sCMOS, and EMCCD, relative to an ideal detector,
// against the signal per pixel.  Lines use physics/detectors.js; dots are
// microsim's detector simulation at the starting parameters
// (detector_data in scripts/make_doc_figures.py).
import * as D from "../physics/detectors.js";
import {
  C,
  Plot,
  baseOptions,
  crosshair,
  decades,
  fmt,
  legendItem,
  loadData,
  responsive,
  shell,
  slider,
  stat,
  tooltipHTML,
  yRangeMark,
} from "./lib.js";

const LEVELS = Array.from({ length: 201 }, (_, i) => 10 ** (-1 + (i * 5) / 200)); // 0.1 to 1e4
const tick = (x) => (x < 1 ? `${x}` : fmt.si(x));

export default async function (el) {
  const data = await loadData("detectors");
  const start = data.cameras;
  const ui = shell(el);
  ui.controls.classList.add("ms-chart__controls--grid");
  const state = {
    ccdRead: start.CCD.read_noise,
    cmosRead: start.sCMOS.read_noise,
    emGain: start.EMCCD.em_gain,
    emRead: start.EMCCD.read_noise,
  };
  const redraw = () => chart.redraw();
  const noise = (label, key) =>
    slider(ui.controls, {
      label,
      min: 0.5,
      max: 100,
      log: true,
      value: state[key],
      format: (v) => `${fmt.sig(v, 2)} e⁻`,
      onInput: (v) => ((state[key] = v), redraw()),
    });
  noise("CCD read noise", "ccdRead");
  noise("sCMOS read noise", "cmosRead");
  slider(ui.controls, {
    label: "EM gain",
    min: 1,
    max: 1000,
    log: true,
    value: state.emGain,
    format: (v) => fmt.int(v),
    onInput: (v) => ((state.emGain = v), redraw()),
  });
  noise("EMCCD read noise", "emRead");

  const color = { CCD: C.series[0], sCMOS: C.series[2], EMCCD: C.series[1] };
  for (const name of Object.keys(color)) legendItem(ui.legend, { label: name, color: color[name] });
  legendItem(ui.legend, { label: "microsim simulation (starting values)", color: C.muted, swatch: "dotmark" });

  const setCross = stat(ui.stats, { label: "EMCCD beats sCMOS", note: "" });
  const setCCD = stat(ui.stats, { label: "CCD reaches 90% of ideal", note: "" });

  const chart = responsive(ui.plot, (width) => {
    const enf = D.emccdExcessNoiseFactor(state.emGain, start.EMCCD.em_stages);
    const model = {
      CCD: { readNoise: state.ccdRead },
      sCMOS: { readNoise: state.cmosRead },
      EMCCD: { readNoise: state.emRead, gain: state.emGain, enf },
    };
    const relative = (name, electrons) =>
      D.snr({ electrons, ...model[name] }) / D.snr({ electrons });
    const rows = LEVELS.flatMap((e) =>
      Object.keys(model).map((name) => ({ name, e, rel: relative(name, e) })),
    );
    // simulated points, only for cameras still at their starting values
    const atStart = {
      CCD: state.ccdRead === start.CCD.read_noise,
      sCMOS: state.cmosRead === start.sCMOS.read_noise,
      EMCCD: state.emGain === start.EMCCD.em_gain && state.emRead === start.EMCCD.read_noise,
    };
    const dots = Object.entries(data.simulated_snr)
      .filter(([name]) => atStart[name])
      .flatMap(([name, pts]) =>
        pts.map((p) => ({ name, e: p.electrons, rel: p.snr / D.snr({ electrons: p.electrons }) })),
      );

    const better = LEVELS.filter((e) => relative("EMCCD", e) > relative("sCMOS", e));
    setCross(
      better.length ? `< ${fmt.sig(better.at(-1), 2)}<small> e⁻</small>` : "never",
      better.length ? "photoelectrons per pixel, with these settings" : "with these settings",
    );
    const good = LEVELS.find((e) => relative("CCD", e) >= 0.9);
    setCCD(good ? `${fmt.sig(good, 2)}<small> e⁻</small>` : "> 10k", "photoelectrons per pixel");

    const svg = Plot.plot({
      ...baseOptions(width),
      height: Math.max(220, Math.min(300, width * 0.4)),
      marginLeft: 44,
      marginBottom: 38,
      x: { type: "log", label: "signal (photoelectrons per pixel)", labelAnchor: "center", labelArrow: "none", labelOffset: 34, ticks: decades(0.1, 1e4), tickFormat: tick },
      y: { domain: [0, 1.02], label: "SNR / ideal SNR", labelArrow: "none", grid: true, tickFormat: ".0%" },
      color: { domain: Object.keys(color), range: Object.values(color) },
      marks: [
        yRangeMark(),
        Plot.ruleY([1], { stroke: C.faint, strokeDasharray: "3,3" }),
        Plot.text([{}], { x: LEVELS[2], y: 1, text: () => "ideal (shot noise only)", dy: 10, textAnchor: "start", fill: C.muted }),
        Plot.lineY(rows, { x: "e", y: "rel", z: "name", stroke: "name", strokeWidth: 2.25 }),
        Plot.dot(dots, { x: "e", y: "rel", fill: "name", r: 3.5, stroke: C.surface, strokeWidth: 1.5 }),
      ],
    });
    ui.plot.replaceChildren(svg);
    crosshair(ui.plot, svg, {
      x: svg.scale("x"),
      content: (e) =>
        tooltipHTML(
          `${fmt.sig(e, 2)} e⁻ per pixel · ideal SNR ${fmt.sig(D.snr({ electrons: e }), 3)}`,
          Object.keys(model).map((name) => ({
            label: name,
            value: `SNR ${fmt.sig(D.snr({ electrons: e, ...model[name] }), 3)} (${fmt.pct(relative(name, e))})`,
            color: color[name],
          })),
        ),
    });
  });
}
