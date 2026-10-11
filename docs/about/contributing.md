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

Most figures are made by microsim when the docs are built, and are not
checked in. Generate them once before previewing (this needs network access
for FPbase and OpenOrganelle, and takes a few minutes the first time):

```bash
uv run --group docs python scripts/make_doc_figures.py              # all figures
uv run --group docs python scripts/make_doc_figures.py psf spectra  # some
```

They are written to `docs/images/generated/`, which is ignored by git. The
figures of the [two-color tutorial](../tutorials/two-color-cell.md) are made by
running the code blocks of that page, so the code shown always matches the
images.
