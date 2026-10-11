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
uv run --group docs zensical build --strict  # broken links and cross-references
```

`test_docs.py` skips pages that need the network when it thinks there is none
(`tests/_util.py` `HAVE_INTERNET`). If it skips, run those pages anyway.
`` ```py `` blocks are fragments and are **not** run: check them by reading.

## What only reading can check

Go page by page through `docs/` (`zensical.toml` `nav` lists them). For each
statement of fact about microsim, find the code that makes it true, and cite it
(`file:line`) in your report. Kinds of claims that go stale, and where they are:

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
  grep -rn '^ *\(!!!\|???\) ' docs
  ```

  A new feature can make a "not modeled" line false; a refactor can make a
  "this is how it works" line false. Check each one.
- **Equations.** The formulas in `guide/photometry.md`, `guide/detectors.md`,
  and `guide/modalities.md` should match the code step by step (normalizations,
  units, which quantity is summed). The JS copies of the same physics are
  covered by the `check-js-physics` skill.
- **Library contents.** Tables of `optical_config.lib` filter sets,
  `detectors.lib` presets, settings, extras (`getting-started/installation.md`),
  and dataset or layer names. Compare with the modules and `pyproject.toml`.
- **Numbers in text and captions.** Figures and charts are regenerated on every
  build, but numbers quoted in the text are not (e.g. "392 photons absorbed",
  "about 6 photons", "71%"). Recompute each one from the current code, or from
  the generated chart data in `docs/data/generated/`.
- **Names.** Every class, method, parameter, and keyword mentioned in prose or
  `` ```py `` fragments must exist with that name and signature. The strict build
  only checks `[name][microsim.x]` cross-references, not plain `` `code` ``.
- **Links to issues.** Statements tied to a GitHub issue (e.g. #152): check
  whether the issue is closed and the statement still holds.

## Reporting and fixing

- Fix clear factual errors directly, in the docs' existing style: plain,
  scientific, no marketing language, no em dashes.
- If the code looks wrong rather than the docs (a doc describes intended behavior
  the code doesn't implement), don't change either; report it.
- Report what you checked, what you changed, and anything you could not verify,
  with `file:line` references for both the docs and the code.
