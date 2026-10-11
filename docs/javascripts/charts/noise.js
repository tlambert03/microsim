// Detectors page: one noise-free image (the ER of a HeLa cell, simulated by
// microsim: noise_data in scripts/make_doc_figures.py) recorded by a camera, with
// the noise drawn in the browser by physics/detectors.js simulateCamera.
import * as D from "../physics/detectors.js";
import { rng } from "../physics/random.js";
import {
  button,
  decodeImage,
  drawImage,
  fmt,
  imagePanel,
  loadData,
  percentiles,
  select,
  shell,
  slider,
  stat,
} from "./lib.js";

// starting values, as in the "Which camera when" chart; `electronsPerAdu` puts
// the signal well inside a 16-bit range.  The offset is 5x the largest read
// noise on the slider (100 e⁻ at 0.5 e⁻/ADU), so the background never clips at 0.
const OFFSET = 1000;
const CAMERAS = {
  "ideal (shot noise only)": { readNoise: 0, electronsPerAdu: 1, offset: 0 },
  CCD: { readNoise: 6, electronsPerAdu: 1, offset: OFFSET },
  sCMOS: { readNoise: 1.5, electronsPerAdu: 0.5, offset: OFFSET },
  EMCCD: { readNoise: 50, electronsPerAdu: 20, emGain: 300, offset: OFFSET },
};

export default async function (el) {
  const { image } = await loadData("noise");
  const truth = decodeImage(image);
  const ui = shell(el);
  ui.controls.classList.add("ms-chart__controls--grid");
  const state = { camera: "CCD", photons: 20, readNoise: 6, emGain: 300, seed: 1 };

  const panels = document.createElement("div");
  panels.className = "ms-panels";
  ui.plot.append(panels);
  const left = imagePanel(panels, "expected photoelectrons (no noise)");
  const right = imagePanel(panels, "");

  select(ui.controls, {
    label: "Camera",
    options: Object.keys(CAMERAS),
    value: state.camera,
    onInput: (v) => {
      state.camera = v;
      state.readNoise = CAMERAS[v].readNoise;
      readSlider.set(state.readNoise);
      update();
    },
  });
  slider(ui.controls, {
    label: "photons at the brightest pixel",
    min: 0.3,
    max: 3000,
    log: true,
    value: state.photons,
    format: (v) => fmt.sig(v, 2),
    onInput: (v) => ((state.photons = v), update()),
  });
  const readSlider = slider(ui.controls, {
    label: "read noise",
    min: 0.1,
    max: 100,
    log: true,
    value: state.readNoise,
    format: (v) => `${fmt.sig(v, 2)} e⁻`,
    onInput: (v) => ((state.readNoise = v), update()),
  });
  const gainSlider = slider(ui.controls, {
    label: "EM gain",
    min: 1,
    max: 1000,
    log: true,
    value: state.emGain,
    format: (v) => fmt.int(v),
    onInput: (v) => ((state.emGain = v), update()),
  });
  const actions = document.createElement("div");
  actions.className = "ms-control ms-control--button";
  ui.controls.append(actions);
  // play: a new noise draw every 100 ms (10 fps), until stopped or the page changes
  const PLAY = "▶ Play noise";
  const STOP = "■ Stop";
  let timer = 0;
  const stop = () => {
    clearInterval(timer);
    timer = 0;
    play.textContent = PLAY;
    play.setAttribute("aria-pressed", "false");
  };
  const play = button(actions, {
    label: PLAY,
    onClick: () => {
      if (timer) return stop();
      play.textContent = STOP;
      play.setAttribute("aria-pressed", "true");
      timer = setInterval(() => {
        if (!el.isConnected) return stop();
        state.seed++;
        update();
      }, 100);
    },
  });
  play.setAttribute("aria-pressed", "false");

  const setSnr = stat(ui.stats, { label: "SNR at the brightest pixel" });
  const setRel = stat(ui.stats, { label: "relative to an ideal detector" });

  function update() {
    const cam = CAMERAS[state.camera];
    const isEM = state.camera === "EMCCD";
    readSlider.input.closest(".ms-control").hidden = state.camera.startsWith("ideal");
    gainSlider.input.closest(".ms-control").hidden = !isEM;
    const gain = isEM ? state.emGain : 1;
    const enf = isEM ? D.emccdExcessNoiseFactor(gain) : 1;
    const readNoise = state.camera.startsWith("ideal") ? 0 : state.readNoise;
    const params = { ...cam, readNoise, gain, enf, bitDepth: 16 };
    // EMCCD: keep the amplified signal inside the ADC range
    if (isEM) params.electronsPerAdu = Math.max(1, gain / 15);

    const expected = truth.data.map((x) => x * state.photons);
    const gray = D.simulateCamera(expected, params, rng(state.seed));
    drawImage(left.canvas, expected, truth.width, truth.height, { hi: state.photons });
    const [lo, hi] = percentiles(gray);
    drawImage(right.canvas, gray, truth.width, truth.height, { lo, hi });
    right.label.textContent = `recorded gray values (${fmt.int(lo)}–${fmt.int(hi)} shown)`;

    const snr = D.snr({ electrons: state.photons, readNoise, gain, enf });
    setSnr(fmt.sig(snr, 3));
    setRel(fmt.pct(snr / D.snr({ electrons: state.photons })));
  }
  update();
}
