---
name: check-js-physics
description: Check that the JavaScript physics behind the docs' interactive charts still matches microsim's Python. Use after changing photometry, detector, or optical-config code in src/microsim, after editing docs/javascripts/physics or docs/javascripts/charts, or when asked to review the docs charts for drift.
---

# Check the docs' JavaScript physics against microsim

The interactive charts in the docs (`docs/javascripts/charts/*.js`) recompute some
of microsim's physics in the browser, so readers can vary parameters freely. That
code is a copy, and copies drift. This skill is how to check it.

## Layout

- `docs/javascripts/physics/*.js`: the copied physics. Pure functions, no DOM, no
  imports. Every function has a comment naming the Python it mirrors.
- `docs/javascripts/charts/*.js`: rendering. These should contain **no physics**:
  they call the physics modules, and only choose parameters, axes, and labels.
- `scripts/make_doc_figures.py`: exports the data the charts load
  (`*_data` functions, written to `docs/data/generated/*.json`): spectra, presets,
  and Monte Carlo results computed by microsim itself.
- `tests/test_js_physics.py` and `tests/js/*.mjs`: run the physics modules in node
  on the same inputs as microsim, and compare. Skipped if `node` is not installed.

## Procedure

1. **Run the tests.** `uv run pytest tests/test_js_physics.py -v`. They must pass,
   and must not be skipped (install node if they are). A failure is drift: decide
   which side is right, and fix the other, or both, together.

2. **Read each mirrored pair.** The tests cover the cases they construct, not every
   branch. For every function in `docs/javascripts/physics/`, open the Python it
   names and compare line by line: constants, normalizations (sum vs. peak vs.
   integral), units (nm vs. m, ns vs. s, W vs. photons), defaults (`|| 1`, `?? null`),
   and how arrays are aligned. Current pairs:

   | JavaScript (`docs/javascripts/physics/`) | Python (`src/microsim/schema/`) |
   | --- | --- |
   | `photometry.js` `mulSpectra` | `spectrum.py` `Spectrum._intensity_op` (outer join, fill 0) |
   | `photometry.js` `mulArrays` | xarray arithmetic (inner join) |
   | `photometry.js` `excitationPath`, `emissionPath` | `optical_config/config.py` `OpticalConfig.excitation`, `.emission`, `._merge` |
   | `photometry.js` `illumination` | `OpticalConfig.illumination`, `LightSource.scaled_spectrum` |
   | `photometry.js` `irradiance`, `fluxDensity` | `OpticalConfig.irradiance`, `.illumination_flux_density` |
   | `photometry.js` `crossSection` | `sample/fluorophore.py` `Fluorophore.absorption_cross_section` |
   | `photometry.js` `absorptionRate`, `saturationParameter`, `totalEmissionRate`, `filteredEmissionRate` | `OpticalConfig` methods of the same names |
   | `photometry.js` `collectionEfficiency` | `lens.py` `ObjectiveLens.collection_efficiency` |
   | `photometry.js` `photonBudget` | `Simulation.filtered_emission_rates` (collection applied after filtering) |
   | `detectors.js` `emccdExcessNoiseFactor` | `detectors/_camera.py` `CameraEMCCD.excess_noise_factor` |
   | `detectors.js` `pmtGain`, `pmtExcessNoiseFactor` | `detectors/_point.py` `PMT.multiplication_gain`, `.excess_noise_factor` |
   | `detectors.js` `hydCountingEfficiency` | `detectors/_point.py` `HyD._counting_efficiency` |
   | `detectors.js` `snr` | derived from `detectors/_detector.py` `_Detector.simulate` and `apply_multiplication_gain` (tested against a Monte Carlo run of `render`) |

   If the Python has gained a step (a new factor, a new parameter that changes the
   result), the JavaScript needs it too, or the chart's caption must say what it
   leaves out.

3. **Find physics outside the physics modules.** In `docs/javascripts/charts/`, look
   for formulas: constants like `1e-9`, `Math.exp`, `Math.sqrt`, `** 2`, `/ (1 +`.
   Ratios of physics outputs for display (e.g. "fraction passed") are fine. Anything
   that computes a physical quantity belongs in `physics/`, with a mirror comment
   and a test.

4. **Check the exported data.** `scripts/make_doc_figures.py` writes presets and
   spectra from microsim objects. Make sure the chart uses the exported values
   (e.g. PMT presets, the HyD dead time), not numbers typed into the JavaScript.
   Starting values typed into a chart (e.g. a default slider position) should match
   the parameters named in the page text or caption.

5. **Check the captions and page text.** Numbers quoted in the docs (e.g. "about 6
   photons in a 100 ms exposure", "71%") should match what the chart shows at its
   starting values. Serve the docs and look:

   ```bash
   uv run --group docs python scripts/make_doc_figures.py photometry_data detector_data
   uv run --group docs zensical serve
   ```

## When adding a chart or a physics function

- Put the physics in `docs/javascripts/physics/`, with a comment naming the Python.
- Add a case to `tests/test_js_physics.py` that compares it with microsim.
- Add a row to the table above.
