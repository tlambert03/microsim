// Run docs/javascripts/physics/detectors.js on cases from tests/test_js_physics.py.
import { readFileSync } from "node:fs";
import * as D from "../../docs/javascripts/physics/detectors.js";

const input = JSON.parse(readFileSync(0, "utf8"));
const out = {
  emccd: input.emccd.map(([g, n]) => D.emccdExcessNoiseFactor(g, n)),
  pmt: input.pmt.map((p) => {
    const gain = D.pmtGain(p);
    return [gain, D.pmtExcessNoiseFactor(gain, p.dynode_stages)];
  }),
  hyd: input.hyd.map(([rate, tau]) => D.hydCountingEfficiency(rate, tau)),
  hyd_max: input.hyd.map(([, tau]) => D.hydMaxRate(tau)),
  round: input.round.map((x) => D.roundHalfEven(x)),
  snr: input.snr.map((p) => D.snr(p)),
};
process.stdout.write(JSON.stringify(out));
