// Photon budget page, saturation section: emission rate against irradiance, with
// and without excitation saturation (OpticalConfig.total_emission_rate).
import * as P from "../physics/photometry.js";
import {
  C,
  Plot,
  baseOptions,
  crosshair,
  fmt,
  legendItem,
  responsive,
  select,
  shell,
  stat,
  tooltipHTML,
  yRangeMark,
} from "./lib.js";
import { loadPhotometry } from "./photometry-data.js";

const POWERS = Array.from({ length: 181 }, (_, i) => 10 ** (-1 + i / 20)); // 0.1 to 1e8
// typical irradiance (see the table in docs/guide/channels.md)
const RANGES = [
  { label: "widefield", x1: 0.1, x2: 10 },
  { label: "confocal spot", x1: 1e4, x2: 1e6 },
];
const tick = (x) => (x < 1 ? `${x}` : fmt.si(x));

export default async function (el) {
  const { fluors, channels } = await loadPhotometry();
  const withLifetime = fluors.filter((f) => f.lifetime_ns);
  const ui = shell(el);
  const state = { fluor: "EGFP", channel: "FITC" };
  const redraw = () => chart.redraw();

  select(ui.controls, {
    label: "Fluorophore",
    options: withLifetime.map((f) => ({ label: `${f.name} (τ = ${f.lifetime_ns} ns)`, value: f.name })),
    value: state.fluor,
    onInput: (v) => ((state.fluor = v), redraw()),
  });
  select(ui.controls, {
    label: "Channel",
    options: channels.map((c) => c.name),
    value: state.channel,
    onInput: (v) => ((state.channel = v), redraw()),
  });
  legendItem(ui.legend, { label: "with saturation", color: C.series[1] });
  legendItem(ui.legend, { label: "linear (no saturation)", color: C.muted, swatch: "dash" });
  legendItem(ui.legend, { label: "kτ = 1", color: C.ink, swatch: "dotmark" });
  const setHalf = stat(ui.stats, {
    label: "irradiance at kτ = 1",
    note: "emission is half the linear value",
  });
  const setMax = stat(ui.stats, { label: "maximum emission rate", note: "QY / τ" });

  const chart = responsive(ui.plot, (width) => {
    const fluor = withLifetime.find((f) => f.name === state.fluor);
    const channel = channels.find((c) => c.name === state.channel);
    // k is proportional to irradiance, so compute it once at 1 W/cm^2
    const k1 = P.total(P.absorptionRate(channel.illumination, 1, fluor));
    const at = (power) => {
      const absorbed = P.absorptionRate(channel.illumination, power, fluor);
      return {
        power,
        k: P.total(absorbed),
        saturated: P.total(P.totalEmissionRate(absorbed, fluor, true)),
        linear: P.total(P.totalEmissionRate(absorbed, fluor, false)),
      };
    };
    const rows = POWERS.map(at);
    const tau = fluor.lifetime_ns * 1e-9;
    const pHalf = 1 / (k1 * tau); // where the saturation parameter k tau is 1
    const qy = fluor.qy || 1;
    const halfRate = at(pHalf).saturated;
    setHalf(`${fmt.si(pHalf)}<small> W/cm²</small>`);
    setMax(`${fmt.si(qy / tau)}<small> photons/s</small>`);

    const yMax = 1e9;
    const svg = Plot.plot({
      ...baseOptions(width),
      height: Math.max(220, Math.min(300, width * 0.38)),
      marginLeft: 52,
      marginBottom: 38,
      x: { type: "log", label: "irradiance (W/cm²)", labelAnchor: "center", labelArrow: "none", labelOffset: 34, tickFormat: tick },
      y: { type: "log", domain: [rows[0].linear / 2, yMax], label: "photons emitted / s", labelArrow: "none", grid: true, tickFormat: "~s" },
      marks: [
        yRangeMark(),
        Plot.rect(RANGES, { x1: "x1", x2: "x2", fill: C.ink, fillOpacity: 0.045 }),
        Plot.text(RANGES, { x: (d) => Math.sqrt(d.x1 * d.x2), frameAnchor: "top", dy: 6, text: "label", fill: C.muted }),
        Plot.lineY(rows, { x: "power", y: "linear", stroke: C.muted, strokeWidth: 1.5, strokeDasharray: "5,4", clip: true }),
        Plot.lineY(rows, { x: "power", y: "saturated", stroke: C.series[1], strokeWidth: 2.5 }),
        Plot.ruleX([pHalf], { stroke: C.faint, strokeDasharray: "2,3" }),
        Plot.dot([{ x: pHalf, y: halfRate }], { x: "x", y: "y", r: 4.5, fill: C.ink, stroke: C.surface, strokeWidth: 2 }),
      ],
    });
    ui.plot.replaceChildren(svg);
    crosshair(ui.plot, svg, {
      x: svg.scale("x"),
      content: (power) => {
        const r = at(power);
        return tooltipHTML(`${fmt.sig(power, 2)} W/cm²`, [
          { label: "with saturation", value: fmt.si(r.saturated), color: C.series[1] },
          { label: "linear", value: fmt.si(r.linear), color: C.muted },
          { label: "kτ", value: fmt.sig(P.saturationParameter(r.k, fluor.lifetime_ns), 2) },
        ]);
      },
    });
  });
}
