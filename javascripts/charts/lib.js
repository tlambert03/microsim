// Shared pieces for the interactive charts: libraries, data loading, controls,
// layout, crosshair tooltips, and wavelength colors.
//
// Colors are CSS variables (docs/stylesheets/extra.css), so charts follow the
// light/dark toggle without redrawing.

// Pinned exactly.  Plot imports this same d3 URL, so d3 is loaded once.
export * as d3 from "https://cdn.jsdelivr.net/npm/d3@7.9.0/+esm";
export * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.17/+esm";
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7.9.0/+esm";

export const C = {
  series: [1, 2, 3, 4, 5].map((i) => `var(--ms-c${i})`),
  ink: "var(--ms-ink)",
  muted: "var(--ms-ink-muted)",
  faint: "var(--ms-ink-faint)",
  grid: "var(--ms-grid)",
  surface: "var(--ms-surface)",
};

// ---------------------------------------------------------------------------
// Data, written by scripts/make_doc_figures.py

const cache = new Map();
export function loadData(name) {
  if (!cache.has(name)) {
    const url = new URL(`../../data/generated/${name}.json`, import.meta.url);
    cache.set(
      name,
      fetch(url).then((r) => {
        if (!r.ok) throw new Error(`${url}: ${r.status}`);
        return r.json();
      }),
    );
  }
  return cache.get(name);
}

// ---------------------------------------------------------------------------
// Layout

function h(tag, className, parent) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (parent) parent.append(el);
  return el;
}

// The parts of a chart, top to bottom: controls, legend, plot, stats.
export function shell(el) {
  el.replaceChildren();
  const controls = h("div", "ms-chart__controls", el);
  const legend = h("div", "ms-chart__legend", el);
  const plot = h("div", "ms-chart__plot", el);
  const stats = h("div", "ms-chart__stats", el);
  return { el, controls, legend, plot, stats };
}

let uid = 0;
const nextId = (prefix) => `${prefix}-${++uid}`;

// A labeled <select>.  `options` are strings, or `{label, value}`.
export function select(parent, { label, options, value, onInput }) {
  const id = nextId("ms-select");
  const wrap = h("div", "ms-control", parent);
  const lab = h("label", "ms-control__label", wrap);
  lab.htmlFor = id;
  lab.textContent = label;
  const sel = h("select", "ms-select", wrap);
  sel.id = id;
  for (const opt of options) {
    const o = h("option", "", sel);
    o.value = typeof opt === "string" ? opt : opt.value;
    o.textContent = typeof opt === "string" ? opt : opt.label;
  }
  sel.value = value;
  sel.addEventListener("input", () => onInput(sel.value));
  return sel;
}

// A labeled range slider with a live value.  With `log`, the slider moves
// through powers of ten between `min` and `max`.
export function slider(parent, { label, min, max, step, value, log, format, onInput }) {
  const id = nextId("ms-slider");
  const wrap = h("div", "ms-control ms-control--slider", parent);
  const top = h("div", "ms-control__top", wrap);
  const lab = h("label", "ms-control__label", top);
  lab.htmlFor = id;
  lab.textContent = label;
  const out = h("output", "ms-control__value", top);
  out.htmlFor = id;
  const input = h("input", "ms-range", wrap);
  input.id = id;
  input.type = "range";
  const toSlider = log ? Math.log10 : (x) => x;
  const fromSlider = log ? (x) => 10 ** x : (x) => x;
  input.min = toSlider(min);
  input.max = toSlider(max);
  input.step = log ? (toSlider(max) - toSlider(min)) / 200 : step ?? "any";
  input.value = toSlider(value);
  const fmt = format ?? ((x) => `${x}`);
  // `exact` shows a value set in code, rather than the slider's rounded position
  const update = (exact) => {
    const v = exact ?? fromSlider(+input.value);
    out.textContent = fmt(v);
    const frac = (input.value - input.min) / (input.max - input.min);
    input.style.setProperty("--fill", `${frac * 100}%`);
    return v;
  };
  input.addEventListener("input", () => onInput(update()));
  update(value);
  return {
    input,
    set(v) {
      input.value = toSlider(v);
      update(v);
    },
  };
}

// A legend entry.  `swatch` is "line", "dash", "area", or "gradient".
export function legendItem(parent, { label, color, swatch = "line" }) {
  const item = h("span", `ms-legend__item ms-legend__item--${swatch}`, parent);
  const sw = h("span", "ms-legend__swatch", item);
  sw.style.setProperty("--swatch", color);
  h("span", "", item).textContent = label;
  return item;
}

// A stat tile: a label and a large value.  Returns a setter.
export function stat(parent, { label, note }) {
  const tile = h("div", "ms-stat", parent);
  const value = h("div", "ms-stat__value", tile);
  h("div", "ms-stat__label", tile).textContent = label;
  const sub = note ? h("div", "ms-stat__note", tile) : null;
  if (sub) sub.textContent = note;
  return (v, n) => {
    value.innerHTML = v;
    if (sub && n !== undefined) sub.textContent = n;
  };
}

// Calls `draw(width)` now and whenever the container's width changes.
export function responsive(container, draw) {
  let width = 0;
  let frame = 0;
  const run = () => {
    const w = Math.floor(container.clientWidth);
    if (w && w !== width) {
      width = w;
      draw(width);
    }
  };
  new ResizeObserver(() => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(run);
  }).observe(container);
  run();
  return {
    redraw() {
      if (width) draw(width);
    },
  };
}

// ---------------------------------------------------------------------------
// Crosshair: a vertical rule following the pointer, and an HTML tooltip whose
// content comes from `content(xValue)`.

export function crosshair(container, svg, { x, content }) {
  const [top, bottom] = svg.__yRange;
  const rule = d3
    .select(svg)
    .append("line")
    .attr("class", "ms-crosshair")
    .attr("y1", top)
    .attr("y2", bottom)
    .style("display", "none");
  let tip = container.querySelector(".ms-tooltip");
  if (!tip) tip = h("div", "ms-tooltip", container);
  tip.hidden = true;
  const [x0, x1] = x.range;

  const move = (event) => {
    const [px] = d3.pointer(event, svg);
    if (px < Math.min(x0, x1) || px > Math.max(x0, x1)) return leave();
    const value = x.invert(px);
    const html = content(value);
    if (!html) return leave();
    rule.attr("x1", px).attr("x2", px).style("display", null);
    tip.innerHTML = html;
    tip.hidden = false;
    // keep the tooltip inside the container, on the side away from the pointer
    const box = container.getBoundingClientRect();
    const svgBox = svg.getBoundingClientRect();
    const left = svgBox.left - box.left + px;
    const flip = left > box.width / 2;
    tip.style.left = `${flip ? left - tip.offsetWidth - 14 : left + 14}px`;
    tip.style.top = `${svgBox.top - box.top + top}px`;
  };
  const leave = () => {
    rule.style("display", "none");
    tip.hidden = true;
  };
  svg.addEventListener("pointermove", move);
  svg.addEventListener("pointerleave", leave);
}

// Tooltip rows: `[{label, value, color}]` under a title.
export function tooltipHTML(title, rows) {
  const body = rows
    .map(
      (r) =>
        `<div class="ms-tooltip__row"><span class="ms-tooltip__key" style="--swatch:${r.color ?? "transparent"}">${r.label}</span><span class="ms-tooltip__val">${r.value}</span></div>`,
    )
    .join("");
  return `<div class="ms-tooltip__title">${title}</div>${body}`;
}

// Records the y range of a rendered plot, for the crosshair.
export function yRangeMark() {
  return (index, scales, values, dimensions, context) => {
    const { marginTop, height, marginBottom } = dimensions;
    context.ownerSVGElement.__yRange = [marginTop, height - marginBottom];
    return null;
  };
}

// ---------------------------------------------------------------------------
// Wavelength colors

// Approximate perceived color of monochromatic light, after Dan Bruton's
// "Approximate RGB values for visible wavelengths".  Outside 380-780 nm the
// color fades to a dim violet or red.
export function wavelengthColor(nm) {
  let r = 0;
  let g = 0;
  let b = 0;
  if (nm < 440) [r, g, b] = [-(nm - 440) / 60, 0, 1];
  else if (nm < 490) [r, g, b] = [0, (nm - 440) / 50, 1];
  else if (nm < 510) [r, g, b] = [0, 1, -(nm - 510) / 20];
  else if (nm < 580) [r, g, b] = [(nm - 510) / 70, 1, 0];
  else if (nm < 645) [r, g, b] = [1, -(nm - 645) / 65, 0];
  else [r, g, b] = [1, 0, 0];
  let f = 1;
  if (nm < 420) f = 0.3 + (0.7 * Math.max(nm - 380, 0)) / 40;
  else if (nm > 700) f = 0.3 + (0.7 * Math.max(780 - nm, 0)) / 80;
  const c = (x) => Math.round(255 * Math.min(1, Math.max(0, x * f)) ** 0.8);
  return `rgb(${c(r)}, ${c(g)}, ${c(b)})`;
}

// A function mark that defines a horizontal gradient of wavelength colors, so
// marks can use `fill: "url(#id)"`.
export function spectrumGradient(id, [lo, hi] = [350, 800]) {
  return (index, scales) => {
    const [x0, x1] = [scales.x(lo), scales.x(hi)];
    const defs = d3.create("svg:defs");
    const grad = defs
      .append("linearGradient")
      .attr("id", id)
      .attr("gradientUnits", "userSpaceOnUse")
      .attr("x1", x0)
      .attr("x2", x1);
    for (let nm = lo; nm <= hi; nm += 10) {
      grad
        .append("stop")
        .attr("offset", (nm - lo) / (hi - lo))
        .attr("stop-color", wavelengthColor(nm));
    }
    return defs.node();
  };
}

// The same colors as a CSS gradient, for HTML legend swatches.
export function cssSpectrum([lo, hi] = [350, 800]) {
  const stops = [];
  for (let nm = lo; nm <= hi; nm += 25) stops.push(wavelengthColor(nm));
  return `linear-gradient(90deg, ${stops.join(", ")})`;
}

export const uniqueId = nextId;

// Standard Plot options shared by the charts.
export function baseOptions(width) {
  return {
    width,
    style: {
      fontFamily: "inherit",
      fontSize: "11px",
      color: C.muted,
      background: "transparent",
      overflow: "visible",
    },
  };
}

// Powers of ten from `lo` to `hi`, for log axis ticks.
export const decades = (lo, hi) =>
  Array.from({ length: Math.round(Math.log10(hi / lo)) + 1 }, (_, i) => lo * 10 ** i);

const SUPERSCRIPT = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };

export const fmt = {
  si: d3.format(".3~s"),
  sig: (x, n = 3) => (x === 0 ? "0" : d3.format(`.${n}~r`)(x)),
  pct: d3.format(".1~%"),
  int: d3.format(",.0f"),
  // `n` significant digits, as `5.89 × 10¹⁶` if |x| < 1e-3 or >= 1e5
  sci: (x, n = 3) => {
    const a = Math.abs(Number(x.toPrecision(n)));
    if (x === 0 || (a >= 1e-3 && a < 1e5)) return fmt.sig(x, n);
    const [m, e] = x.toExponential(n - 1).split("e");
    const exp = String(Number(e)).replace(/[-0-9]/g, (c) => SUPERSCRIPT[c]);
    return `${Number(m)} × 10${exp}`;
  },
};

// ---------------------------------------------------------------------------
// Images

// Inverse of `image_json` in scripts/make_doc_figures.py: values 0-1.
export function decodeImage({ shape, data }) {
  const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
  const u16 = new Uint16Array(bytes.buffer);
  return { width: shape[1], height: shape[0], data: Float32Array.from(u16, (x) => x / 65535) };
}

// 256-entry RGB lookup tables
export const LUT = {
  gray: Array.from({ length: 256 }, (_, i) => [i, i, i]),
  magma: Array.from({ length: 256 }, (_, i) => {
    const c = d3.rgb(d3.interpolateMagma(i / 255));
    return [c.r, c.g, c.b];
  }),
};

// The `lo`-th and `hi`-th percentiles of `values`, for display contrast.
export function percentiles(values, lo = 0.001, hi = 0.999) {
  const s = Float64Array.from(values).sort();
  return [s[Math.floor(lo * (s.length - 1))], s[Math.floor(hi * (s.length - 1))]];
}

// Draws `values` (width x height, row-major) on `canvas`, mapping [lo, hi] through
// `gamma` and `lut`.
export function drawImage(canvas, values, width, height, { lo = 0, hi = 1, gamma = 1, lut = LUT.gray } = {}) {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(width, height);
  const span = hi - lo || 1;
  for (let i = 0; i < width * height; i++) {
    const t = Math.min(Math.max((values[i] - lo) / span, 0), 1) ** gamma;
    const [r, g, b] = lut[Math.round(t * 255)];
    img.data[4 * i] = r;
    img.data[4 * i + 1] = g;
    img.data[4 * i + 2] = b;
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// A labeled canvas panel.
export function imagePanel(parent, label) {
  const fig = document.createElement("div");
  fig.className = "ms-panel";
  const canvas = document.createElement("canvas");
  const cap = document.createElement("div");
  cap.className = "ms-panel__label";
  cap.textContent = label;
  fig.append(canvas, cap);
  parent.append(fig);
  return { canvas, label: cap };
}

// A plain button.
export function button(parent, { label, onClick }) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "ms-button";
  b.textContent = label;
  b.addEventListener("click", onClick);
  parent.append(b);
  return b;
}
