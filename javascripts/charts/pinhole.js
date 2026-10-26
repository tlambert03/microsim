// Modalities page: widefield and confocal PSFs across pinhole sizes, all computed
// by microsim (pinhole_data in scripts/make_doc_figures.py).  No physics here.
import {
  C,
  LUT,
  Plot,
  baseOptions,
  crosshair,
  decodeImage,
  drawImage,
  fmt,
  imagePanel,
  legendItem,
  loadData,
  responsive,
  shell,
  slider,
  stat,
  tooltipHTML,
  yRangeMark,
} from "./lib.js";

export default async function (el) {
  const data = await loadData("pinhole");
  const psfs = data.psfs.map((p) => ({ ...p, xy: decodeImage(p.xy), xz: decodeImage(p.xz) }));
  const widefield = psfs.find((p) => p.pinhole_au === null);
  const confocal = psfs.filter((p) => p.pinhole_au !== null);
  const ui = shell(el);
  const state = { i: confocal.findIndex((p) => p.pinhole_au === 1) };

  slider(ui.controls, {
    label: "pinhole_au",
    min: 0,
    max: confocal.length - 1,
    step: 1,
    value: state.i,
    format: (i) => `${confocal[Math.round(i)].pinhole_au} AU`,
    onInput: (v) => ((state.i = Math.round(v)), update()),
  });
  const gammaSel = document.createElement("label");
  gammaSel.className = "ms-control ms-check";
  gammaSel.innerHTML = `<input type="checkbox" checked> square-root display`;
  ui.controls.append(gammaSel);
  const sqrt = gammaSel.querySelector("input");
  sqrt.addEventListener("input", () => update());
  const dimSel = document.createElement("label");
  dimSel.className = "ms-control ms-check";
  dimSel.innerHTML = `<input type="checkbox"> show the signal lost at the pinhole`;
  ui.controls.append(dimSel);
  const dim = dimSel.querySelector("input");
  dim.addEventListener("input", () => update());

  const panels = document.createElement("div");
  panels.className = "ms-panels";
  panels.style.setProperty("--cols", 4);
  ui.legend.after(panels);
  const views = {
    wfXY: imagePanel(panels, "widefield · XY"),
    wfXZ: imagePanel(panels, "widefield · XZ"),
    cfXY: imagePanel(panels, ""),
    cfXZ: imagePanel(panels, ""),
  };
  for (const v of Object.values(views)) v.canvas.parentElement.classList.add("ms-panel--smooth");

  const size = `${fmt.sig(data.image_dx_um * (widefield.xy.width - 1), 2)} µm`;
  const note = document.createElement("span");
  note.className = "ms-legend__note";
  note.textContent = `images ${size} square, z vertical in XZ`;
  ui.legend.append(note);
  legendItem(ui.legend, { label: "widefield", color: C.muted, swatch: "dash" });
  legendItem(ui.legend, { label: "confocal", color: C.series[1] });

  const setThru = stat(ui.stats, {
    label: "light through the pinhole",
    note: "for a point at the focus, of the light the objective collects",
  });
  const setLat = stat(ui.stats, { label: "lateral FWHM" });
  const setAx = stat(ui.stats, { label: "axial FWHM" });

  // `brightness` scales the image, which is normalized to its own peak
  const draw = (view, img, brightness = 1) =>
    drawImage(view.canvas, img.data, img.width, img.height, {
      hi: 1 / brightness,
      gamma: sqrt.checked ? 0.5 : 1,
      lut: LUT.magma,
    });
  const zOf = (p) => p.plane_sums.map((v, i) => ({ z: (i - Math.floor(p.plane_sums.length / 2)) * data.dz_um, v }));

  const chart = responsive(ui.plot, (width) => {
    const cur = confocal[state.i];
    const w = width > 520 ? Math.floor((width - 24) / 2) : width;
    const sectioning = Plot.plot({
      ...baseOptions(w),
      height: 200,
      marginLeft: 44,
      marginBottom: 38,
      x: { label: "distance from focus (µm)", labelAnchor: "center", labelArrow: "none", labelOffset: 34 },
      y: { domain: [0, 1.05], label: "signal from a thin plane", labelArrow: "none", grid: true, tickFormat: ".0%" },
      marks: [
        yRangeMark(),
        Plot.lineY(zOf(widefield), { x: "z", y: "v", stroke: C.muted, strokeWidth: 1.5, strokeDasharray: "5,4" }),
        Plot.lineY(zOf(cur), { x: "z", y: "v", stroke: C.series[1], strokeWidth: 2.25 }),
      ],
    });
    const rows = confocal.flatMap((p) => [
      { au: p.pinhole_au, nm: p.fwhm_lateral_nm, kind: "lateral" },
      { au: p.pinhole_au, nm: p.fwhm_axial_nm, kind: "axial" },
    ]);
    const resolution = Plot.plot({
      ...baseOptions(w),
      height: 200,
      marginLeft: 44,
      marginBottom: 38,
      x: { label: "pinhole (AU)", labelAnchor: "center", labelArrow: "none", labelOffset: 34 },
      y: { domain: [0, 560], label: "FWHM (nm)", labelArrow: "none", grid: true },
      marks: [
        Plot.ruleY([widefield.fwhm_lateral_nm, widefield.fwhm_axial_nm], { stroke: C.muted, strokeDasharray: "5,4" }),
        Plot.lineY(rows, { x: "au", y: "nm", z: "kind", stroke: C.series[1], strokeWidth: 2 }),
        Plot.text(rows.filter((r) => r.au === 3), { x: "au", y: "nm", text: "kind", dy: -9, textAnchor: "end", fill: C.muted }),
        Plot.dot(rows.filter((r) => r.au === cur.pinhole_au), { x: "au", y: "nm", r: 4.5, fill: C.series[1], stroke: C.surface, strokeWidth: 2 }),
      ],
    });
    const box = document.createElement("div");
    box.className = "ms-chart__panels";
    box.append(sectioning, resolution);
    ui.plot.replaceChildren(box);
    crosshair(ui.plot, sectioning, {
      x: sectioning.scale("x"),
      content: (z) => {
        const at = (p) => zOf(p).reduce((a, b) => (Math.abs(b.z - z) < Math.abs(a.z - z) ? b : a));
        return tooltipHTML(`${fmt.sig(at(cur).z, 2)} µm from focus`, [
          { label: "widefield", value: fmt.pct(at(widefield).v), color: C.muted },
          { label: `confocal, ${cur.pinhole_au} AU`, value: fmt.pct(at(cur).v), color: C.series[1] },
        ]);
      },
    });
  });

  function update() {
    const cur = confocal[state.i];
    draw(views.wfXY, widefield.xy);
    draw(views.wfXZ, widefield.xz);
    // with `dim`, the confocal peak is the fraction of collected light that passes
    // the pinhole (1 for an open pinhole), as computed by microsim
    const brightness = dim.checked ? cur.peak : 1;
    draw(views.cfXY, cur.xy, brightness);
    draw(views.cfXZ, cur.xz, brightness);
    views.cfXY.label.textContent = `confocal ${cur.pinhole_au} AU · XY`;
    views.cfXZ.label.textContent = `confocal ${cur.pinhole_au} AU · XZ`;
    setThru(fmt.pct(cur.peak));
    setLat(`${fmt.int(cur.fwhm_lateral_nm)}<small> nm (widefield ${fmt.int(widefield.fwhm_lateral_nm)})</small>`);
    setAx(`${fmt.int(cur.fwhm_axial_nm)}<small> nm (widefield ${fmt.int(widefield.fwhm_axial_nm)})</small>`);
    chart.redraw();
  }
  update();
}
