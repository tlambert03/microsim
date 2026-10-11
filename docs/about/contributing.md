---
icon: lucide/git-pull-request
---

# Contributing

Bug reports, questions, and pull requests are welcome on
[GitHub](https://github.com/tlambert03/microsim). Contributions of new ground
truth generators, modalities, and detector models are particularly useful.

## Development setup

```bash
git clone https://github.com/tlambert03/microsim
cd microsim
uv sync
uv run pytest
```

Code style is enforced with [ruff](https://docs.astral.sh/ruff/) and type
checking with mypy, run through [prek](https://github.com/j178/prek) (or
pre-commit):

```bash
uv run prek run -a
```

## Documentation

The docs are built with [Zensical](https://zensical.org) from the Markdown
files in `docs/`, configured in `zensical.toml`. The API reference is generated
from docstrings with [mkdocstrings](https://mkdocstrings.github.io).

```bash
uv run --group docs zensical serve   # live preview at http://localhost:8000
uv run --group docs zensical build   # build to site/
```

Most figures, and the data for the interactive charts, are made by microsim
when the docs are built, and are not checked in. Generate them once before
previewing (this needs network access for FPbase and OpenOrganelle, and takes a
few minutes the first time):

```bash
uv run --group docs python scripts/make_doc_figures.py                  # everything
uv run --group docs python scripts/make_doc_figures.py psf detector_data  # some
```

Images are written to `docs/images/generated/` and chart data to
`docs/data/generated/`, both ignored by git. The figures of the
[two-color tutorial](../tutorials/two-color-cell.md) are made by running the
code blocks of that page, so the code shown always matches the images.

### Interactive charts

The interactive charts are drawn in the browser with
[Observable Plot](https://observablehq.com/plot/) and
[D3](https://d3js.org), loaded from jsDelivr. A chart is a placeholder in the
Markdown,

```html
<div class="ms-chart" data-chart="spectra"></div>
```

and a module in `docs/javascripts/charts/` with the same name. Where a chart
recomputes microsim's physics (so readers can change parameters), the
calculation lives in `docs/javascripts/physics/`, and
`tests/test_js_physics.py` runs it in node and compares it with microsim. Run
that test after changing photometry or detector code; it needs `node`.
