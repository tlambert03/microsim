// Photon budget: a JavaScript copy of microsim's photometry, for the docs' charts.
//
// Every function names the Python it mirrors.  If you change one side, change the
// other, and run `uv run pytest tests/test_js_physics.py`.  See
// .claude/skills/check-js-physics/SKILL.md.
//
// A spectrum is `{w: number[], v: number[]}`: wavelengths in nm (ascending) and
// values.  Spectra from microsim are exported by scripts/make_doc_figures.py.

// scipy.constants, as used in src/microsim/schema/optical_config/config.py
export const PLANCK = 6.62607015e-34; // J s
export const LIGHT_SPEED = 299792458; // m/s
export const AVOGADRO = 6.02214076e23; // 1/mol

const sum = (a) => a.reduce((s, x) => s + x, 0);

// Inverse of `spectrum_json` in scripts/make_doc_figures.py
export function decodeSpectrum(s) {
  if (s.w) return { w: s.w, v: s.v };
  return { w: s.v.map((_, i) => s.w0 + i * s.dw), v: s.v };
}

// Align two spectra on their exact wavelengths, the union (`outer`) or the
// intersection (`inner`), filling missing values with 0.
function align(a, b, join) {
  const ma = new Map(a.w.map((w, i) => [w, a.v[i]]));
  const mb = new Map(b.w.map((w, i) => [w, b.v[i]]));
  let w;
  if (join === "outer") w = [...new Set([...a.w, ...b.w])].sort((x, y) => x - y);
  else w = a.w.filter((x) => mb.has(x));
  return { w, va: w.map((x) => ma.get(x) ?? 0), vb: w.map((x) => mb.get(x) ?? 0) };
}

// Spectrum * Spectrum: src/microsim/schema/spectrum.py `Spectrum._intensity_op`
// (xr.align(join="outer"), then fillna(0))
export function mulSpectra(a, b) {
  const { w, va, vb } = align(a, b, "outer");
  return { w, v: va.map((x, i) => x * vb[i]) };
}

// DataArray * DataArray: xarray's default arithmetic join is "inner"
export function mulArrays(a, b) {
  const { w, va, vb } = align(a, b, "inner");
  return { w, v: va.map((x, i) => x * vb[i]) };
}

export const scale = (s, k) => ({ w: s.w, v: s.v.map((x) => x * k) });
export const total = (s) => sum(s.v);

// Spectrum.inverted
export const inverted = (s) => ({ w: s.w, v: s.v.map((x) => 1 - x) });

// OpticalConfig.excitation / OpticalConfig.emission, and OpticalConfig._merge.
// `filters` are `{placement, spectrum}`, placement one of EX, EM, BS, BSi, ALL.
export function excitationPath(filters) {
  const parts = [];
  for (const f of filters) {
    if (["EX", "BSi", "ALL"].includes(f.placement)) parts.push(f.spectrum);
    if (f.placement === "BS") parts.push(inverted(f.spectrum));
  }
  return parts.length ? parts.reduce(mulSpectra) : null;
}

export function emissionPath(filters) {
  const parts = [];
  for (const f of filters) {
    if (["EM", "BS", "ALL"].includes(f.placement)) parts.push(f.spectrum);
    if (f.placement === "BSi") parts.push(inverted(f.spectrum));
  }
  return parts.length ? parts.reduce(mulSpectra) : null;
}

// OpticalConfig.illumination.  `lights` are `{spectrum, power}` (power relative,
// optional): LightSource.scaled_spectrum, summed with Spectrum.__add__ (outer join).
export function illumination(lights, exPath) {
  if (!lights.length) return exPath;
  let spec = null;
  for (const l of lights) {
    const s = l.power == null ? l.spectrum : scale(l.spectrum, l.power);
    if (spec === null) spec = s;
    else {
      const { w, va, vb } = align(spec, s, "outer");
      spec = { w, v: va.map((x, i) => x + vb[i]) };
    }
  }
  return exPath ? mulSpectra(spec, exPath) : spec;
}

// OpticalConfig.irradiance: illumination normalized to sum 1, times `power`
// (W/cm^2; not scaled if power is null, i.e. 1 W/cm^2 in total)
export function irradiance(illum, power) {
  const s = scale(illum, 1 / total(illum));
  return power == null ? s : scale(s, power);
}

// OpticalConfig.illumination_flux_density: W/cm^2 -> photons/cm^2/s, E = h c / λ
export function fluxDensity(irrad) {
  return {
    w: irrad.w,
    v: irrad.v.map((x, i) => x / ((PLANCK * LIGHT_SPEED) / (irrad.w[i] * 1e-9))),
  };
}

// Fluorophore.absorption_cross_section: excitation spectrum normalized to peak 1,
// times the extinction coefficient (M^-1 cm^-1), then ln(10) * 1e3 * ε / N_A (cm^2)
export function crossSection(exSpectrum, extinctionCoefficient) {
  const peak = Math.max(...exSpectrum.v);
  return {
    w: exSpectrum.w,
    v: exSpectrum.v.map(
      (x) => (Math.log(10) * 1e3 * ((x / peak) * extinctionCoefficient)) / AVOGADRO,
    ),
  };
}

// OpticalConfig.absorption_rate: photons/s absorbed per molecule, per wavelength
export function absorptionRate(illum, power, fluor) {
  return mulArrays(fluxDensity(irradiance(illum, power)), crossSection(fluor.ex, fluor.ec));
}

// OpticalConfig.saturation_parameter: k * tau, 0 if the lifetime is unknown
export function saturationParameter(totalAbsorption, lifetimeNs) {
  if (!lifetimeNs) return 0;
  return totalAbsorption * lifetimeNs * 1e-9;
}

// OpticalConfig.total_emission_rate: emission spectrum normalized to sum 1, times
// QY (1 if unknown) and the total absorption rate, divided by (1 + k tau) if
// `saturate`
export function totalEmissionRate(absRate, fluor, saturate = true) {
  let k = total(absRate);
  if (saturate) k = k / (1 + saturationParameter(k, fluor.lifetime_ns));
  const norm = total(fluor.em);
  return scale(fluor.em, ((fluor.qy || 1) * k) / norm);
}

// OpticalConfig.filtered_emission_rate: emission rate times the emission path,
// itself multiplied by the detector QE (a number, a spectrum, or null)
export function filteredEmissionRate(emRate, emPath, qe = null) {
  if (emPath === null) return emRate;
  let path = emPath;
  if (typeof qe === "number") path = scale(path, qe);
  else if (qe) path = mulSpectra(path, qe);
  return mulArrays(emRate, path);
}

// ObjectiveLens.collection_efficiency: solid-angle fraction of isotropic emission,
// measured in the specimen (n sin θ is conserved)
export function collectionEfficiency(na, specimenRI) {
  const theta = Math.asin(Math.min(na / specimenRI, 1));
  return (1 - Math.cos(theta)) / 2;
}

// The full chain for one fluorophore in one channel:
// OpticalConfig.filtered_emission_rate, times the collection efficiency as in
// Simulation.filtered_emission_rates.
export function photonBudget({ channel, fluor, power, qe, na, specimenRI, saturate = true }) {
  const absorbed = absorptionRate(channel.illumination, power, fluor);
  const emitted = totalEmissionRate(absorbed, fluor, saturate);
  const filtered = filteredEmissionRate(emitted, channel.emission, qe);
  const collection = collectionEfficiency(na, specimenRI);
  const collected = scale(filtered, collection);
  return {
    absorbed,
    emitted,
    filtered,
    collected,
    collection,
    saturation: saturationParameter(total(absorbed), fluor.lifetime_ns),
  };
}
