// Run docs/javascripts/physics/photometry.js on cases from tests/test_js_physics.py.
// Reads `{fluorophores, channels, cases}` as JSON on stdin, writes results to stdout.
import { readFileSync } from "node:fs";
import * as P from "../../docs/javascripts/physics/photometry.js";

const input = JSON.parse(readFileSync(0, "utf8"));
const totals = (s) => ({ w: s.w, v: s.v, total: P.total(s) });

const out = input.cases.map((c) => {
  const fluor = input.fluorophores[c.fluor];
  const ch = input.channels[c.channel];
  const exPath = P.excitationPath(ch.filters);
  const channel = {
    illumination: P.illumination(ch.lights, exPath),
    emission: P.emissionPath(ch.filters),
  };
  const b = P.photonBudget({ channel, fluor, ...c.params });
  const k1 = P.total(P.absorptionRate(channel.illumination, 1, fluor));
  return {
    cross_section: totals(P.crossSection(fluor.ex, fluor.ec)),
    illumination_flux_density: totals(
      P.fluxDensity(P.irradiance(channel.illumination, c.params.power)),
    ),
    absorbed: totals(b.absorbed),
    emitted: totals(b.emitted),
    filtered: totals(b.filtered),
    collected: totals(b.collected),
    collection: b.collection,
    saturation: b.saturation,
    half_saturation_irradiance: fluor.lifetime_ns
      ? P.halfSaturationIrradiance(k1, fluor.lifetime_ns)
      : null,
    max_emission_rate: fluor.lifetime_ns ? P.maxEmissionRate(fluor) : null,
  };
});
process.stdout.write(JSON.stringify(out));
