// Detector models: a JavaScript copy of parts of microsim's detectors, for the
// docs' charts.
//
// Every function names the Python it mirrors.  If you change one side, change the
// other, and run `uv run pytest tests/test_js_physics.py`.  See
// .claude/skills/check-js-physics/SKILL.md.

import * as R from "./random.js";

// CameraEMCCD.excess_noise_factor (src/microsim/schema/detectors/_camera.py):
// F = 2 (G - 1) G^(-(N + 1) / N) + 1 / G, for mean gain G over N stages
export function emccdExcessNoiseFactor(gain, stages = 604) {
  return 2 * (gain - 1) * gain ** (-(stages + 1) / stages) + 1 / gain;
}

// PMT.multiplication_gain (src/microsim/schema/detectors/_point.py):
// `hv_gain` if set, otherwise ref_gain * (hv / ref_hv) ** hv_exponent
export function pmtGain({ hv = null, ref_hv = 0.8, ref_gain = 5e5, hv_exponent = 6.7, hv_gain = null }) {
  if (hv_gain != null) return hv_gain;
  const v = hv == null ? ref_hv : hv;
  return ref_gain * (v / ref_hv) ** hv_exponent;
}

// PMT.excess_noise_factor: sum of delta^-i for i = 0..n, with delta the gain per
// dynode, gain^(1/n)
export function pmtExcessNoiseFactor(gain, dynodeStages = 9) {
  const delta = gain ** (1 / dynodeStages);
  let f = 0;
  for (let i = 0; i <= dynodeStages; i++) f += delta ** -i;
  return f;
}

// HyD._counting_efficiency: non-paralyzable dead time, a true rate n (counts/s)
// is recorded as n / (1 + n tau)
export function hydCountingEfficiency(rate, deadTimeNs = 1.5) {
  return 1 / (1 + rate * deadTimeNs * 1e-9);
}

// Signal-to-noise ratio of one pixel, from the steps of _Detector.simulate
// (src/microsim/schema/detectors/_detector.py), ignoring quantization, clipping,
// and dead time:
//   shot noise: Poisson, variance = mean electrons (signal + dark)
//   multiplication: variance x F (apply_multiplication_gain); read noise, added
//     after the gain G, is G times smaller relative to the signal
//   averaging N passes: signal and dark integrate N times as long, read noise is
//     added N times (_read_noise_per_pass, std read_noise * sqrt(N))
// `electrons` is the mean number of photoelectrons per pass (QE already applied,
// as in microsim, where QE is part of the optical image).
export function snr({ electrons, dark = 0, readNoise = 0, gain = 1, enf = 1, averaging = 1 }) {
  const signal = electrons * averaging;
  const shot = enf * (signal + dark * averaging);
  const read = (readNoise ** 2 * averaging) / gain ** 2;
  return signal / Math.sqrt(shot + read);
}

// One frame from a camera, following _Detector.simulate step by step, for an image
// of `expected` mean photoelectrons per pixel (QE and exposure already applied).
// Not modeled here: averaging, binning, dark current, and clock-induced charge.
//   1-2. shot noise: Poisson
//   3. full well: clip at `fullWell` electrons
//   4. multiplication (apply_multiplication_gain): Gamma(n / (F - 1), G (F - 1)),
//      rounded; exactly n * G if F = 1
//   5. readout: Gaussian read noise, / electrons_per_adu, round, clip to the bit
//      depth, + offset, clip at 0 and at the maximum
export function simulateCamera(expected, p, rand) {
  const { readNoise = 6, gain = 1, enf = 1, electronsPerAdu = 1, offset = 100 } = p;
  const { bitDepth = 16, fullWell = Infinity } = p;
  const maxGray = 2 ** bitDepth - 1;
  const out = new Float64Array(expected.length);
  for (let i = 0; i < expected.length; i++) {
    let e = R.poisson(rand, Math.max(expected[i], 0));
    e = Math.min(e, fullWell);
    if (gain !== 1) {
      if (enf - 1 < 1e-9) e = Math.round(e * gain);
      else if (e > 0) e = Math.round(R.gamma(rand, e / (enf - 1), gain * (enf - 1)));
    }
    const voltage = e + readNoise * R.normal(rand);
    const signal = Math.min(Math.round(voltage / electronsPerAdu), maxGray);
    out[i] = Math.min(Math.max(signal + offset, 0), maxGray);
  }
  return out;
}
