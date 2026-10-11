// Channels page: a fluorophore's spectra against a channel's excitation light and
// emission path, with the parts that get through shaded.
import * as P from "../physics/photometry.js";
import { loadPhotometry } from "./photometry-data.js";
import {
  C,
  Plot,
  baseOptions,
  cssSpectrum,
  crosshair,
  d3,
  fmt,
  legendItem,
  responsive,
  select,
  shell,
  spectrumGradient,
  stat,
  tooltipHTML,
  uniqueId,
  wavelengthColor,
  yRangeMark,
} from "./lib.js";

const DOMAIN = [350, 800];

const peakNormalized = (s) => P.scale(s, 1 / Math.max(...s.v));
const toRows = (s) => s.w.map((w, i) => ({ w, v: s.v[i] }));

function valueAt(s, nm) {
  if (!s) return null;
  const i = d3.bisector((d) => d).center(s.w, nm);
  return Math.abs(s.w[i] - nm) <= 1 ? s.v[i] : 0;
}

// Absorption relative to having all excitation photons at the absorption peak.
function excitationEfficiency(channel, fluor) {
  const absorbed = P.total(P.absorptionRate(channel.illumination, 1, fluor));
  const flux = P.total(P.fluxDensity(P.irradiance(channel.illumination, 1)));
  const peak = Math.max(...P.crossSection(fluor.ex, fluor.ec).v);
  return absorbed / (flux * peak);
}

// Fraction of emitted photons that pass the emission path.
function emissionFraction(channel, fluor) {
  const em = P.scale(fluor.em, 1 / P.total(fluor.em));
  return P.total(P.filteredEmissionRate(em, channel.emission));
}

export default async function (el) {
  const { fluors, channels } = await loadPhotometry();
  const ui = shell(el);
  const state = {
    fluor: el.dataset.fluor ?? "EGFP",
    channel: el.dataset.channel ?? "FITC",
  };

  select(ui.controls, {
    label: "Fluorophore",
    options: fluors.map((f) => f.name),
    value: state.fluor,
    onInput: (v) => ((state.fluor = v), chart.redraw()),
  });
  select(ui.controls, {
    label: "Channel",
    options: channels.map((c) => c.name),
    value: state.channel,
    onInput: (v) => ((state.channel = v), chart.redraw()),
  });

  const gid = uniqueId("ms-spectrum");
  const fill = `url(#${gid})`;
  const rainbow = cssSpectrum(DOMAIN);
  legendItem(ui.legend, { label: "absorption", color: rainbow, swatch: "dash" });
  legendItem(ui.legend, { label: "emission", color: rainbow, swatch: "line" });
  legendItem(ui.legend, { label: "excitation light", color: C.muted, swatch: "dot" });
  legendItem(ui.legend, { label: "emission path", color: C.ink, swatch: "line" });
  legendItem(ui.legend, { label: "absorbed / detected", color: rainbow, swatch: "area" });

  const setEx = stat(ui.stats, {
    label: "excitation efficiency",
    note: "absorption relative to all light at the absorption peak",
  });
  const setEm = stat(ui.stats, {
    label: "emission passed",
    note: "fraction of emitted photons through the emission path",
  });
  const setRel = stat(ui.stats, {
    label: "relative signal",
    note: "ε × QY × both efficiencies, relative to EGFP in FITC",
  });
  const ref = (() => {
    const f = fluors.find((x) => x.name === "EGFP");
    const c = channels.find((x) => x.name === "FITC");
    return f.ec * f.qy * excitationEfficiency(c, f) * emissionFraction(c, f);
  })();

  const chart = responsive(ui.plot, (width) => {
    const fluor = fluors.find((f) => f.name === state.fluor);
    const channel = channels.find((c) => c.name === state.channel);
    const ex = peakNormalized(fluor.ex);
    const em = peakNormalized(fluor.em);
    const light = peakNormalized(channel.illumination);
    const path = channel.emission;
    const absorbed = P.mulArrays(ex, light);
    const detected = path ? P.mulArrays(em, path) : em;

    const exEff = excitationEfficiency(channel, fluor);
    const emFrac = emissionFraction(channel, fluor);
    setEx(fmt.pct(exEff));
    setEm(fmt.pct(emFrac));
    setRel(fmt.sig((fluor.ec * (fluor.qy || 1) * exEff * emFrac) / ref, 2) + "×");

    const svg = Plot.plot({
      ...baseOptions(width),
      height: Math.max(240, Math.min(340, width * 0.42)),
      marginLeft: 40,
      marginBottom: 38,
      x: { domain: DOMAIN, label: "wavelength (nm)", labelAnchor: "center", labelArrow: "none", labelOffset: 34 },
      y: { domain: [0, 1.05], label: null, grid: true, ticks: 5, tickFormat: d3.format(".0%") },
      marks: [
        spectrumGradient(gid, DOMAIN),
        yRangeMark(),
        Plot.areaY(toRows(ex), { x: "w", y: "v", fill, fillOpacity: 0.1, clip: true }),
        Plot.areaY(toRows(em), { x: "w", y: "v", fill, fillOpacity: 0.12, clip: true }),
        Plot.areaY(toRows(absorbed), { x: "w", y: "v", fill, fillOpacity: 0.75, clip: true }),
        Plot.areaY(toRows(detected), { x: "w", y: "v", fill, fillOpacity: 0.75, clip: true }),
        Plot.lineY(toRows(ex), { x: "w", y: "v", stroke: fill, strokeWidth: 2, strokeDasharray: "5,3", clip: true }),
        Plot.lineY(toRows(em), { x: "w", y: "v", stroke: fill, strokeWidth: 2, clip: true }),
        Plot.lineY(toRows(light), { x: "w", y: "v", stroke: C.muted, strokeWidth: 1.5, strokeDasharray: "1.5,3", strokeLinecap: "round", clip: true }),
        path ? Plot.lineY(toRows(path), { x: "w", y: "v", stroke: C.ink, strokeWidth: 1.5, clip: true }) : null,
        Plot.ruleY([0], { stroke: C.faint }),
      ],
    });
    ui.plot.replaceChildren(svg);
    crosshair(ui.plot, svg, {
      x: svg.scale("x"),
      content: (nm) =>
        tooltipHTML(`${Math.round(nm)} nm`, [
          { label: "absorption", value: fmt.pct(valueAt(ex, nm)), color: wavelengthColor(nm) },
          { label: "emission", value: fmt.pct(valueAt(em, nm)), color: wavelengthColor(nm) },
          { label: "excitation light", value: fmt.pct(valueAt(light, nm)), color: C.muted },
          { label: "emission path", value: path ? fmt.pct(valueAt(path, nm)) : "–", color: C.ink },
        ]),
    });
  });
}
