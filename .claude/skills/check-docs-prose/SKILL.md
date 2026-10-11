---
name: check-docs-prose
description: Check that the hand-written text of the microsim docs (docs/**/*.md) is still true of the code - defaults, required parameters, what each stage uses, what is and isn't modeled, numbers in captions, code examples. Use after changes to src/microsim, before a release, or when asked to audit the docs for stale or wrong statements.
---

# Check the docs' prose against the code

The guide pages are written by hand, and microsim changes quickly. Statements
like "Only `truth_space` and `sample` are required", "the default is 1 W/cm²",
or "sCMOS pixel-to-pixel variation is not modeled" go stale silently. This skill
is how to find them.

Do not trust the docs, or your memory of the code: check every claim against the
source, or by running it.

## What the build already checks

Run these first; fix what they find.

```bash
uv run pytest tests/test_docs.py        # runs every ```python block in the docs
uv run pytest tests/test_js_physics.py  # JS chart physics (see check-js-physics)
uv run --group docs python scripts/make_doc_figures.py   # figures, chart data
uv run python scripts/make_notebooks.py
uv run --group docs zensical build --strict --clean  # links and cross-references
```

- `make_doc_figures.py` takes minutes and needs network access (FPbase,
  OpenOrganelle, Allen). It only writes gitignored files, except
  `cosem_examples`, which is only regenerated when named. It is also the only
  thing that runs `tutorials/two-color-cell.md` (through the `tutorials`
  figure), and it prints values that captions quote (e.g. `first_simulation`).
- `test_docs.py` never runs `two-color-cell.md` (its `SKIP` set). Without
  network access it silently *drops* the pages in `NEEDS_INTERNET` rather than
  skipping them. Check what ran with
  `uv run pytest tests/test_docs.py --collect-only -q` (one item per page, plus
  `README.md`), and run dropped pages another way.
- Without `--clean`, the build can be incremental and miss problems.

## What only reading can check

Go page by page through `docs/` (`zensical.toml` `nav` lists them). For each
statement of fact about microsim, find the code that makes it true, and cite it
(`file:line`) in your report. Kinds of claims that go stale, and where they are:

- **Code that isn't tested.** `` ```py `` blocks are fragments and are **not**
  run, ```` ```json title=... ```` blocks are written out but never validated,
  and shell commands in the docs (`about/contributing.md`,
  `getting-started/installation.md`) are never run. Run the cheap,
  self-contained ones in a scratch script (I/O, small spaces), validate JSON
  examples with `ms.Simulation.model_validate_json`, and check
  `make_doc_figures.py` arguments against its `@figure` function names.
- **Defaults and required fields.** Tables of defaults
  (`getting-started/first-simulation.md`, `guide/settings.md`,
  `guide/detectors.md` "Default camera settings", `guide/channels.md`), and
  sentences like "Only `truth_space` and `sample` are required"
  (`guide/index.md`). Check the pydantic field definitions, including
  `default_factory` and validators that fill values in.
- **What each stage uses.** The stage diagrams (the mermaid blocks in
  `index.md` and `guide/index.md`) and the stage table in `guide/index.md` list
  which parameters feed which method. Check `Simulation.ground_truth`,
  `filtered_emission_rates`, `optical_image`, and `digital_image`, and what they
  call.
- **What is and isn't modeled.** `guide/index.md` "What is (and isn't)
  modeled", and every admonition that states a limitation or a workaround:

  ```bash
  grep -rn '^ *\(!!!\|???\) ' docs | grep -v 'docs/\(data\|notebooks\)'
  ```

  A new feature can make a "not modeled" line false; a refactor can make a
  "this is how it works" line false. Check each one. Also check that each
  setting the docs describe actually takes effect (e.g. that
  `max_psf_radius_aus` changes the PSF size in the log line `Need PSF (nz,nx)`).
- **Equations.** The formulas in `guide/photometry.md`, `guide/detectors.md`,
  and `guide/modalities.md` should match the code step by step (normalizations,
  units, which quantity is summed, the order of steps). The JS copies of the
  same physics are covered by the `check-js-physics` skill.
- **Library contents.** Tables of `optical_config.lib` filter sets,
  `detectors.lib` presets, settings, extras (`getting-started/installation.md`),
  and dataset or layer names. Compare with the modules and `pyproject.toml`.
  Figure captions describe what `scripts/make_doc_figures.py` computed (e.g. the
  shared `_cell_sim()`): check them against that script.
- **Numbers in text and captions.** Figures and charts are regenerated on every
  build, but numbers quoted in the text are not (e.g. "392 photons absorbed",
  "about 6 photons", "71%"). Recompute each one from the current code, from the
  output of `make_doc_figures.py`, or from the generated chart data in
  `docs/data/generated/`.
- **Names.** Every class, method, parameter, and keyword mentioned in prose or
  `` ```py `` fragments must exist with that name and signature. The strict build
  only checks `[name][microsim.x]` cross-references, not plain `` `code` ``.
- **New API.** Public methods and settings added since the docs were written
  (`git log` on `src/microsim`) should be mentioned where they belong (e.g. a
  new `Simulation` method in `guide/index.md` and `api/index.md`).
- **Links to issues.** Statements tied to a GitHub issue (e.g. #152): check
  whether the issue is closed and the statement still holds.

## Style

The docs are plain and scientific. Check these with a search, then by reading:

```bash
grep -rn '—' docs --include='*.md' | grep -v 'docs/\(data\|notebooks\)'   # em dashes
grep -rniE '\b(recommend|best for|ideal for|natural choice|should use|which .* when|beats|outperform)' \
  docs --include='*.md' docs/javascripts/charts | grep -v 'docs/\(data\|notebooks\)'
```

- No em dashes, and no marketing language.
- The docs describe what microsim computes; they don't advise readers on which
  hardware to use (e.g. "which camera when", "the natural choice", "best for
  low light"). Rephrase such lines as statements of what the model or the
  physics does. Advice on using microsim itself (e.g. "Prefer Zarr") is fine.
- Chart labels and captions in `docs/javascripts/charts/*.js` count too.

## Reporting and fixing

- Fix clear factual errors directly, in the docs' existing style, unless you
  were asked for a report only.
- If the code looks wrong rather than the docs (a doc describes intended behavior
  the code doesn't implement), don't change either; report it.
- Report what you checked, what you changed, and anything you could not verify,
  with `file:line` references for both the docs and the code.
