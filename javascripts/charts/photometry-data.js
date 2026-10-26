// Fluorophores, channels, detector QE, and the default objective lens from
// photometry.json, decoded, with each channel's illumination and emission path
// computed as microsim does.
import * as P from "../physics/photometry.js";
import { loadData } from "./lib.js";

let prepared = null;

export function loadPhotometry() {
  prepared ??= loadData("photometry").then(prepare);
  return prepared;
}

function prepare(data) {
  const fluors = data.fluorophores.map((f) => ({
    ...f,
    ex: P.decodeSpectrum(f.ex),
    em: P.decodeSpectrum(f.em),
  }));
  const channels = data.channels.map((c) => {
    const filters = c.filters.map((f) => ({ ...f, spectrum: P.decodeSpectrum(f.spectrum) }));
    const lights = c.lights.map((l) => ({ ...l, spectrum: P.decodeSpectrum(l.spectrum) }));
    return {
      name: c.name,
      filters,
      lights,
      illumination: P.illumination(lights, P.excitationPath(filters)),
      emission: P.emissionPath(filters),
    };
  });
  const detectors = data.detectors.map((d) => ({
    ...d,
    qe: typeof d.qe === "number" ? d.qe : P.decodeSpectrum(d.qe),
  }));
  return { fluors, channels, detectors, lens: data.lens };
}

