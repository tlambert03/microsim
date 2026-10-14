import os
import re
from pathlib import Path
from textwrap import dedent

import pytest

from ._util import HAVE_INTERNET

DOCS = Path(__file__).parent.parent / "docs"
DOCS_MDS = list(DOCS.rglob("*.md"))
DOCS_MDS += [Path(__file__).parent.parent / "README.md"]
CODE_BLOCK = re.compile(r"```python([^`]*)```", re.DOTALL)
JSON_BLOCK = re.compile(r"```json([^`]*)```", re.DOTALL)

# Only ```python blocks are executed; ```py blocks are illustrative fragments.
# pages that download large datasets (OpenOrganelle) are not run
SKIP = {"two-color-cell.md"}
# pages that fetch fluorophores or optical configs from FPbase
NEEDS_INTERNET = {
    "point-scanning-detectors.md",
    "samples.md",
    "your-own-ground-truth.md",
}
DOCS_MDS = [p for p in DOCS_MDS if p.name not in SKIP]
if not HAVE_INTERNET:
    DOCS_MDS = [p for p in DOCS_MDS if p.name not in NEEDS_INTERNET]


@pytest.mark.parametrize("doc", DOCS_MDS, ids=lambda p: str(p.relative_to(DOCS.parent)))
def test_docs(doc: Path, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    os.chdir(tmp_path)
    source = doc.read_text(encoding="utf-8")

    if jsons := [dedent(match.group(1)) for match in JSON_BLOCK.finditer(source)]:
        # any json block with a title is written to a file in the test directory
        for json_block in jsons:
            l0, *lines = json_block.splitlines()
            if l0.strip().startswith("title"):
                title = l0.split("=")[1].strip("'\"").strip()
                Path(title).write_text("\n".join(lines))
                lines = [x[3:] for x in lines]

    blocks = [dedent(match.group(1)) for match in CODE_BLOCK.finditer(source)]
    # share a single namespace across blocks so names (e.g. imports) from earlier
    # blocks stay visible in later ones. Bare `exec(block)` used to leak through
    # the function's locals, but PEP 667 ended that on py3.13+.
    namespace: dict[str, object] = {}
    for n, block in enumerate(blocks):
        lines = [x for x in block.splitlines() if not x.lstrip().startswith("#")]
        block = "\n".join(lines)
        try:
            exec(block, namespace)
        except Exception as e:
            raise ValueError(
                f"Error in docs file {doc}\n"
                f"Code block #{n} cannot be executed as written:\n"
                f"{block}\n"
                f"\n--------------------------\n{e}"
            ) from e
