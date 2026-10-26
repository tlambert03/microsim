---
name: check-js-physics
description: Check that the JavaScript physics behind the docs' interactive charts still matches microsim's Python. Use after changing photometry, detector, or optical-config code in src/microsim, after editing docs/javascripts/physics or docs/javascripts/charts, or when asked to review the docs charts for drift.
---

# Check the docs' JavaScript physics against microsim

The interactive charts in the docs (`docs/javascripts/charts/*.js`) recompute some
of microsim's physics in the browser, so readers can vary parameters freely. That
code is a copy, and copies drift. This skill is how to check it.

## Layout

- `docs/javascripts/physics/*.js`: the copied physics. Pure functions, no DOM, and
  no imports from outside `physics/`. Every function that mirrors microsim has a
  comment naming the Python it mirrors (helpers like `scale`, `total`, and `rng`
  don't).
- `docs/javascripts/charts/*.js`: rendering. These should contain **no physics**:
  they call the physics modules, and only choose parameters, axes, and labels.
  `lib.js` (plot helpers), `index.js` (mounting), and `photometry-data.js`
  (decodes `photometry.json`) are infrastructure.
- `scripts/make_doc_figures.py`: exports the data the charts load, written to
  `docs/data/generated/` (gitignored): `photometry_data` → `photometry.json`,
  `detector_data` → `detectors.json`, `noise_data` → `noise.json`,
  `pinhole_data` → `pinhole.json`. Spectra, presets, default lens values, and
  images computed by microsim itself.
- `tests/test_js_physics.py` and `tests/js/*.mjs`: run the physics modules in node
  on the same inputs as microsim, and compare. Skipped if `node` is not installed.

### The charts

| Chart | Page | Physics used | Data | Starting values |
| --- | --- | --- | --- | --- |
| `spectra.js` | `guide/channels.md` | `absorptionRate`, `crossSection`, `filteredEmissionRate`, `fluxDensity`, `irradiance`, `transmitted` | `photometry.json` | fluorophore and channel menus |
| `crosstalk.js` | `guide/channels.md` | `photonBudget` | `photometry.json` (incl. `lens`) | EGFP + tdTomato, FITC 500 LP + TRITC; numbers in the caption |
| `photon-budget.js` | `guide/photometry.md` | `photonBudget` | `photometry.json` (incl. `lens`) | EGFP, FITC, 1 W/cm², 100 ms; "about 6 photons" in the page text |
| `saturation.js` | `guide/photometry.md` | `absorptionRate`, `totalEmissionRate`, `halfSaturationIrradiance`, `maxEmissionRate` | `photometry.json` | irradiance bands from the table in `guide/channels.md` |
| `noise.js` | `guide/detectors.md` | `simulateCamera`, `snr`, `emccdExcessNoiseFactor` | `noise.json`, `detectors.json` (`cameras`) | camera presets shared with `detector-snr.js` |
| `detector-snr.js` | `guide/detectors.md` | `snr`, `emccdExcessNoiseFactor` | `detectors.json` (`cameras`) | ≈71% cap in the caption |
| `emccd-enf.js` | `guide/detectors.md` | `emccdExcessNoiseFactor` | `detectors.json` (`cameras.EMCCD`) | |
| `pmt-gain.js` | `guide/detectors.md` | `pmtGain`, `pmtExcessNoiseFactor` | `detectors.json` (`pmts`) | F ≈ 1.3 / 1.5 in the page text |
| `hyd-dead-time.js` | `guide/detectors.md` | `hydCountingEfficiency`, `hydMaxRate` | `detectors.json` | "about 6% low at 40 Mcps" in the page text |
| `pinhole.js` | `guide/modalities.md` | none (displays microsim PSFs) | `pinhole.json` | |

Update this table when you add, rename, or remove a chart.

## Procedure

1. **Run the tests.** `uv run pytest tests/test_js_physics.py -v`. They must pass,
   and must not be skipped (install node if they are). A failure is drift: decide
   which side is right, and fix the other, or both, together.

2. **Read each mirrored pair.** The tests cover the cases they construct, not every
   branch. For every function in `docs/javascripts/physics/`, open the Python it
   names and compare line by line: constants, normalizations (sum vs. peak vs.
   integral), units (nm vs. m, ns vs. s, W vs. photons), defaults (`|| 1`, `?? null`),
   how arrays are aligned, and rounding (`np.round` rounds halves to even, unlike
   `Math.round`; use `roundHalfEven`). Current pairs:

   | JavaScript (`docs/javascripts/physics/`) | Python (`src/microsim/` unless noted) |
   | --- | --- |
   | `photometry.js` `decodeSpectrum` | `scripts/make_doc_figures.py` `spectrum_json` (inverse) |
   | `photometry.js` `at` | `schema/spectrum.py` `Spectrum.at` (linear interpolation, 0 outside) |
   | `photometry.js` `mulSpectra` | `schema/spectrum.py` `Spectrum._intensity_op` for `*` (union of wavelengths, both interpolated) |
   | `photometry.js` `transmitted` | `schema/spectrum.py` `Spectrum.transmitted` |
   | `photometry.js` `inverted` | `schema/spectrum.py` `Spectrum.inverted` |
   | `photometry.js` `excitationPath`, `emissionPath` | `schema/optical_config/config.py` `OpticalConfig.excitation`, `.emission`, `._merge` |
   | `photometry.js` `illumination` | `OpticalConfig.illumination`, `LightSource.scaled_spectrum` |
   | `photometry.js` `irradiance`, `fluxDensity` | `OpticalConfig.irradiance`, `.illumination_flux_density` |
   | `photometry.js` `crossSection` | `schema/sample/fluorophore.py` `Fluorophore.absorption_cross_section` |
   | `photometry.js` `absorptionRate`, `saturationParameter`, `totalEmissionRate`, `filteredEmissionRate` | `OpticalConfig` methods of the same names (which call `optical_config/_fluorescence.py` `ChannelFluorescence` and `OpticalConfig.detected_fraction`) |
   | `photometry.js` `halfSaturationIrradiance`, `maxEmissionRate` | derived from `OpticalConfig.saturation_parameter` and `.total_emission_rate` |
   | `photometry.js` `collectionEfficiency` | `schema/lens.py` `ObjectiveLens.collection_efficiency` |
   | `photometry.js` `photonBudget` | `Simulation.filtered_emission_rates` (tested end to end) |
   | `detectors.js` `emccdExcessNoiseFactor` | `schema/detectors/_camera.py` `CameraEMCCD.excess_noise_factor` |
   | `detectors.js` `pmtGain`, `pmtExcessNoiseFactor` | `schema/detectors/_point.py` `PMT.multiplication_gain`, `.excess_noise_factor` |
   | `detectors.js` `hydCountingEfficiency`, `hydMaxRate` | `schema/detectors/_point.py` `HyD._counting_efficiency` |
   | `detectors.js` `roundHalfEven` | `np.round`, as used in `_Detector.simulate` |
   | `detectors.js` `snr` | derived from `schema/detectors/_detector.py` `_Detector.simulate` and `apply_multiplication_gain` (tested against a Monte Carlo run of `render`) |
   | `detectors.js` `simulateCamera` | `_Detector.simulate` steps 1-5 for cameras, and `apply_multiplication_gain` (tested statistically against `render`: mean, variance, quantiles) |
   | `random.js` `poisson`, `normal`, `gamma` | the distributions of `schema/backend.py` `poisson_rvs`, `norm_rvs`, and `scipy.stats.gamma` (same distributions, not the same random streams) |
   | `random.js` `rng` | no mirror (seeded PRNG for the browser) |

   Charts that only display arrays computed by microsim (`pinhole.js`, the noise-free
   image in `noise.js`) have nothing to mirror; check instead that the export
   function in `scripts/make_doc_figures.py` uses microsim, and that captions
   describe what was computed.

   If the Python has gained a step (a new factor, a new parameter that changes the
   result), the JavaScript needs it too, or the function's comment (and the chart's
   caption, if it matters there) must say what it leaves out.

3. **Find physics outside the physics modules.** Search the charts for formulas:

   ```bash
   grep -nE '1e-?[0-9]|Math\.(exp|sqrt|log|pow)|\*\* ?-?[0-9]|/ ?\(1 \+' docs/javascripts/charts/*.js
   ```

   Ratios of physics outputs for display (e.g. "fraction passed") are fine.
   Anything that computes a physical quantity belongs in `physics/`, with a mirror
   comment and a test.

4. **Check the exported data.** `scripts/make_doc_figures.py` writes presets and
   spectra from microsim objects. Make sure the chart uses the exported values
   (e.g. PMT presets, the HyD dead time, the `ObjectiveLens` defaults), not numbers
   typed into the JavaScript. If two charts share starting values (e.g. the camera
   presets of `noise.js` and `detector-snr.js`), both must read them from the same
   export. Display-only settings (slider ranges, `electronsPerAdu` chosen to fit
   16 bits) can stay in the chart.

5. **Check the captions and page text.** Numbers quoted in the docs (see the
   "Starting values" column above) should match what the chart shows at its
   starting values. Regenerate the data first; without it, a chart shows "This
   interactive chart failed to load":

   ```bash
   # noise_data needs network access (FPbase, OpenOrganelle) the first time
   uv run --group docs python scripts/make_doc_figures.py photometry_data detector_data noise_data pinhole_data
   ```

   Then either compute the numbers in node, importing `docs/javascripts/physics/*.js`
   with the generated JSON (fastest), or serve the docs and look:

   ```bash
   uv run --group docs zensical serve
   ```

## When adding a chart or a physics function

- Put the physics in `docs/javascripts/physics/`, with a comment naming the Python.
- Add a case to `tests/test_js_physics.py` that compares it with microsim.
- Add rows to the tables above.
