// Moments of docs/javascripts/physics/detectors.js simulateCamera on flat images,
// for tests/test_js_physics.py.
import { readFileSync } from "node:fs";
import * as D from "../../docs/javascripts/physics/detectors.js";
import { rng } from "../../docs/javascripts/physics/random.js";

const quantile = (sorted, q) => sorted[Math.floor(q * (sorted.length - 1))];
const input = JSON.parse(readFileSync(0, "utf8"));
const out = input.map(({ electrons, params, size }, i) => {
  const img = D.simulateCamera(new Float64Array(size).fill(electrons), params, rng(i + 1));
  const mean = img.reduce((s, x) => s + x, 0) / size;
  const variance = img.reduce((s, x) => s + (x - mean) ** 2, 0) / size;
  const sorted = Float64Array.from(img).sort();
  return { mean, var: variance, quantiles: [0.05, 0.5, 0.95].map((q) => quantile(sorted, q)) };
});
process.stdout.write(JSON.stringify(out));
