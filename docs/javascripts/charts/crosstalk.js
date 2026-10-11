// Channels page, bleedthrough section: detected photons from two fluorophores in
// two channels, per molecule (OpticalConfig.filtered_emission_rate and the
// collection efficiency, via physics/photometry.js), and their emission spectra
// against each channel's emission path.
import * as P from "../physics/photometry.js";
import {
  C,
  Plot,
  baseOptions,
  crosshair,
  cssSpectrum,
  d3,
  fmt,
  legendItem,
  responsive,
  select,
  shell,
  spectrumGradient,
  tooltipHTML,
  uniqueId,
  wavelengthColor,
  yRangeMark,
} from "./lib.js";
import { loadPhotometry } from "./photometry-data.js";

const DOMAIN = [400, 750];
const toRows = (s) => s.w.map((w, i) => ({ w, v: s.v[i] }));
const peakNormalized = (s) => P.scale(s, 1 / Math.max(...s.v));

function valueAt(s, nm) {
  if (!s) return null;
  const i = d3.bisector((d) => d).center(s.w, nm);
  return Math.abs(s.w[i] - nm) <= 1 ? s.v[i] : 0;
}

// detected photons/s per molecule, at 1 W/cm², 1.4 NA, specimen RI 1.47, QE 1
function detected(channel, fluor) {
  const b = P.photonBudget({ channel, fluor, power: 1, qe: null, na: 1.4, specimenRI: 1.47 });
  return P.total(b.collected);
}

export default async function (el) {
  const { fluors, channels } = await loadPhotometry();
  const ui = shell(el);
  const state = {
    f1: "EGFP",
    f2: "tdTomato",
    c1: "FITC, 500 LP emission",
    c2: "TRITC",
  };
  const redraw = () => chart.redraw();
  const pick = (label, key, options) =>
    select(ui.controls, {
      label,
      options,
      value: state[key],
      onInput: (v) => ((state[key] = v), redraw()),
    });
  pick("Fluorophore 1", "f1", fluors.map((f) => f.name));
  pick("Fluorophore 2", "f2", fluors.map((f) => f.name));
  pick("Channel 1", "c1", channels.map((c) => c.name));
  pick("Channel 2", "c2", channels.map((c) => c.name));

  const gid = uniqueId("ms-crosstalk");
  const fill = `url(#${gid})`;
  const rainbow = cssSpectrum(DOMAIN);
  legendItem(ui.legend, { label: "fluorophore 1 emission", color: rainbow, swatch: "area" });
  legendItem(ui.legend, { label: "fluorophore 2 emission", color: rainbow, swatch: "dash" });
  legendItem(ui.legend, { label: "channel 1 emission path", color: C.ink });
  legendItem(ui.legend, { label: "channel 2 emission path", color: C.muted, swatch: "dot" });

  const table = document.createElement("table");
  table.className = "ms-matrix";
  ui.stats.append(table);

  const chart = responsive(ui.plot, (width) => {
    const f = [state.f1, state.f2].map((n) => fluors.find((x) => x.name === n));
    const c = [state.c1, state.c2].map((n) => channels.find((x) => x.name === n));
    const rates = c.map((ch) => f.map((fl) => detected(ch, fl)));

    const head = `<tr><th></th>${f.map((fl) => `<th>${fl.name}</th>`).join("")}</tr>`;
    const body = c
      .map((ch, i) => {
        const tot = rates[i][0] + rates[i][1];
        const cells = rates[i]
          .map((r, j) => {
            const share = tot ? r / tot : 0;
            const cross = share < 0.5 ? " ms-matrix__cross" : "";
            return `<td class="${cross}"><span class="ms-matrix__value">${fmt.sig(r, 3)}</span><span class="ms-matrix__share">${share > 0 && share < 0.001 ? "<0.1%" : fmt.pct(share)} of the channel</span></td>`;
          })
          .join("");
        return `<tr><th>${ch.name}</th>${cells}</tr>`;
      })
      .join("");
    table.innerHTML = `<caption>detected photons/s per molecule (1 W/cm², 1.4 NA, QE 1)</caption>${head}${body}`;

    const em = f.map((fl) => peakNormalized(fl.em));
    const paths = c.map((ch) => ch.emission);
    const svg = Plot.plot({
      ...baseOptions(width),
      height: Math.max(200, Math.min(280, width * 0.36)),
      marginLeft: 40,
      marginBottom: 38,
      x: { domain: DOMAIN, label: "wavelength (nm)", labelAnchor: "center", labelArrow: "none", labelOffset: 34 },
      y: { domain: [0, 1.05], label: null, grid: true, ticks: 5, tickFormat: d3.format(".0%") },
      marks: [
        spectrumGradient(gid, DOMAIN),
        yRangeMark(),
        Plot.areaY(toRows(em[0]), { x: "w", y: "v", fill, fillOpacity: 0.45, clip: true }),
        Plot.lineY(toRows(em[0]), { x: "w", y: "v", stroke: fill, strokeWidth: 2, clip: true }),
        Plot.areaY(toRows(em[1]), { x: "w", y: "v", fill, fillOpacity: 0.12, clip: true }),
        Plot.lineY(toRows(em[1]), { x: "w", y: "v", stroke: fill, strokeWidth: 2, strokeDasharray: "5,3", clip: true }),
        paths[0] ? Plot.lineY(toRows(paths[0]), { x: "w", y: "v", stroke: C.ink, strokeWidth: 1.5, clip: true }) : null,
        paths[1] ? Plot.lineY(toRows(paths[1]), { x: "w", y: "v", stroke: C.muted, strokeWidth: 1.5, strokeDasharray: "1.5,3", strokeLinecap: "round", clip: true }) : null,
      ],
    });
    ui.plot.replaceChildren(svg);
    crosshair(ui.plot, svg, {
      x: svg.scale("x"),
      content: (nm) =>
        tooltipHTML(`${Math.round(nm)} nm`, [
          { label: `${f[0].name} emission`, value: fmt.pct(valueAt(em[0], nm)), color: wavelengthColor(nm) },
          { label: `${f[1].name} emission`, value: fmt.pct(valueAt(em[1], nm)), color: wavelengthColor(nm) },
          { label: c[0].name, value: paths[0] ? fmt.pct(valueAt(paths[0], nm)) : "1", color: C.ink },
          { label: c[1].name, value: paths[1] ? fmt.pct(valueAt(paths[1], nm)) : "1", color: C.muted },
        ]),
    });
  });
}
