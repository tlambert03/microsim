// Random number generators for docs/javascripts/physics/detectors.js.  Same
// distributions as the scipy/numpy samplers microsim uses
// (src/microsim/schema/backend.py: poisson_rvs, norm_rvs; scipy.stats.gamma in
// detectors/_detector.py apply_multiplication_gain), not the same streams.

// mulberry32: a small, fast, seedable uniform generator on [0, 1)
export function rng(seed = 0) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// standard normal, Box-Muller
export function normal(rand) {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

// log(k!) via Stirling's series, exact table for small k
const LOG_FACT = [0, 0, 0.6931471805599453, 1.791759469228055, 3.1780538303479458,
  4.787491742782046, 6.579251212010101, 8.525161361065415, 10.60460290274525,
  12.801827480081469];
function logFactorial(k) {
  if (k < LOG_FACT.length) return LOG_FACT[k];
  const x = k + 1;
  return (x - 0.5) * Math.log(x) - x + 0.5 * Math.log(2 * Math.PI) + 1 / (12 * x) - 1 / (360 * x ** 3);
}

// Poisson: multiplication for small means, PTRS (Hörmann 1993) for large, as numpy
export function poisson(rand, lam) {
  if (lam <= 0) return 0;
  if (lam < 10) {
    const l = Math.exp(-lam);
    let k = 0;
    let p = rand();
    while (p > l) {
      k++;
      p *= rand();
    }
    return k;
  }
  const slam = Math.sqrt(lam);
  const loglam = Math.log(lam);
  const b = 0.931 + 2.53 * slam;
  const a = -0.059 + 0.02483 * b;
  const invalpha = 1.1239 + 1.1328 / (b - 3.4);
  const vr = 0.9277 - 3.6224 / (b - 2);
  for (;;) {
    const u = rand() - 0.5;
    const v = rand();
    const us = 0.5 - Math.abs(u);
    const k = Math.floor(((2 * a) / us + b) * u + lam + 0.43);
    if (us >= 0.07 && v <= vr) return k;
    if (k < 0 || (us < 0.013 && v > us)) continue;
    if (Math.log(v) + Math.log(invalpha) - Math.log(a / (us * us) + b) <= -lam + k * loglam - logFactorial(k)) {
      return k;
    }
  }
}

// Gamma(shape, scale): Marsaglia & Tsang 2000, boosted for shape < 1
export function gamma(rand, shape, scale = 1) {
  if (shape < 1) return gamma(rand, shape + 1, scale) * rand() ** (1 / shape);
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x;
    let v;
    do {
      x = normal(rand);
      v = 1 + c * x;
    } while (v <= 0);
    v = v ** 3;
    const u = rand();
    if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
      return d * v * scale;
    }
  }
}
