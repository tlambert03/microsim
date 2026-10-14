"""Convert the tutorial pages in `docs/tutorials/` to Jupyter notebooks.

Each ```python block becomes a code cell, and the text between them markdown
cells, with docs-only syntax (admonitions, captions, cross-references) turned into
plain Markdown and links made absolute.  A first code cell installs microsim at
the commit the docs were built from, if it isn't installed (as on Google Colab).

Output goes to `docs/notebooks/`, which the docs publish, so the tutorial pages
can link to them, and Colab can open them from the `gh-pages` branch.

Usage:

    uv run python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import subprocess
from pathlib import Path
from textwrap import dedent

DOCS = Path(__file__).parents[1] / "docs"
OUT = DOCS / "notebooks"
SITE = "https://talleylambert.com/microsim/"
REPO = "https://github.com/tlambert03/microsim"

INSTALL = """\
# Install microsim, if it isn't already (e.g. on Google Colab).
# This notebook was made from the docs at commit {sha}. microsim changes
# quickly, so other versions may give errors or different results.
import importlib.util

if importlib.util.find_spec("microsim") is None:
    %pip install -q "{requirement}"
"""


def _commit() -> str:
    if sha := os.environ.get("GITHUB_SHA"):
        return sha
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=DOCS, text=True
    ).strip()


def _url(page: str, link: str) -> str:
    """Absolute URL of `link`, relative to the docs page `page`."""
    if re.match(r"[a-z]+://", link) or link.startswith("#"):
        return link
    path, _, anchor = link.partition("#")
    path = posixpath.normpath(posixpath.join(posixpath.dirname(page), path))
    if path.endswith(".md"):
        path = path[: -len(".md")]
        path = "" if path == "index" else path.removesuffix("/index") + "/"
    return SITE + path + (f"#{anchor}" if anchor else "")


def _admonition(match: re.Match) -> str:
    title = match.group("title") or match.group("kind").capitalize()
    body = dedent(match.group("body")).strip()
    quoted = "\n".join(f"> {line}" if line else ">" for line in body.splitlines())
    return f"> **{title}**\n>\n{quoted}\n"


def _markdown(text: str, page: str) -> str:
    # images made by running this page: the notebook makes them itself
    text = re.sub(
        r"!\[[^\]]*\]\([^)]*generated/tutorials/[^)]*\)(\{[^}]*\})?\n", "", text
    )
    # the "Open in Colab" and "Download notebook" buttons
    text = re.sub(r"^\[.*\]\(.*\)\{[^}]*\.md-button[^}]*\}\n", "", text, flags=re.M)
    text = re.sub(r"/// caption\n(.*?)\n///", r"*\1*", text, flags=re.DOTALL)
    text = re.sub(
        r"^(?:!!!|\?\?\?\+?) (?P<kind>\w+)(?: \"(?P<title>[^\"]*)\")?\n\n"
        r"(?P<body>(?:(?: {4}.*)?\n)+)",
        _admonition,
        text,
        flags=re.MULTILINE,
    )
    # mkdocstrings cross-references: [`name`][microsim.name] -> `name`
    text = re.sub(r"\[([^\]]+)\]\[[\w.]+\]", r"\1", text)
    text = re.sub(
        r"(!?\[[^\]]*\])\(([^)\s]+)\)", lambda m: f"{m[1]}({_url(page, m[2])})", text
    )
    text = re.sub(r"\{ ?[.#][^}]*\}", "", text)  # attribute lists
    # math: \( \) and \[ \] -> $ and $$
    text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text)
    text = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _code(source: str) -> str:
    # code annotations, `# (1)!`, are explained in a list after the code
    return re.sub(r"# \((\d+)\)!", r"# (\1)", dedent(source)).strip()


def _cell(kind: str, source: str, n: int) -> dict:
    cell: dict = {
        "cell_type": kind,
        "id": f"cell-{n}",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def notebook(page: Path, sha: str) -> dict:
    """The notebook for one tutorial page."""
    text = page.read_text()
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)  # front matter
    rel = page.relative_to(DOCS).as_posix()
    extras = "cosem" if "CosemLabel" in text else ""
    requirement = f"microsim{f'[{extras}]' if extras else ''} @ git+{REPO}@{sha}"

    parts = re.split(r"^```python\n(.*?)^```\n", text, flags=re.DOTALL | re.MULTILINE)
    title, _, intro = parts[0].strip().partition("\n")
    web = _url(rel, posixpath.basename(rel))
    header = (
        f"{title}\n\n*From the [microsim documentation]({web}). Run the cells in "
        f"order.*\n\n{_markdown(intro, rel)}"
    )
    cells = [
        _cell("markdown", header, 0),
        _cell("code", INSTALL.format(sha=sha[:12], requirement=requirement), 1),
    ]
    annotated = False
    for i, part in enumerate(parts[1:], start=1):
        if i % 2:  # code
            annotated = bool(re.search(r"# \(\d+\)!", part))
            cells.append(_cell("code", _code(part), len(cells)))
        elif md := _markdown(part, rel):
            if annotated:
                md = f"Notes on the code above:\n\n{md}"
            cells.append(_cell("markdown", md, len(cells)))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    sha = _commit()
    OUT.mkdir(exist_ok=True)
    for page in sorted((DOCS / "tutorials").glob("*.md")):
        if page.stem == "index":
            continue
        path = OUT / f"{page.stem}.ipynb"
        path.write_text(json.dumps(notebook(page, sha), indent=1, ensure_ascii=False))
        print("wrote", path.relative_to(DOCS.parent))
