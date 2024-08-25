import os
import re
from pathlib import Path
from textwrap import dedent

import pytest

DOCS = Path(__file__).parent.parent / "docs"
DOCS_MDS = list(DOCS.rglob("*.md"))
DOCS_MDS += [Path(__file__).parent.parent / "README.md"]
CODE_BLOCK = re.compile(r"```python([^`]*)```", re.DOTALL)
JSON_BLOCK = re.compile(r"```json([^`]*)```", re.DOTALL)


@pytest.mark.parametrize("doc", DOCS_MDS, ids=lambda p: p.name)
def test_docs(doc: Path, tmp_path: Path) -> None:
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
    for n, block in enumerate(blocks):
        lines = [x for x in block.splitlines() if not x.lstrip().startswith("#")]
        if doc.name == "stages.md":
            lines = _fill_in_stages(lines)
        block = "\n".join(lines)
        try:
            exec(block)
        except Exception as e:
            raise ValueError(
                f"Error in docs file {doc}\n"
                f"Code block #{n} cannot be executed as written:\n"
                f"{block}\n"
                f"\n--------------------------\n{e}"
            ) from e


def _fill_in_stages(lines: list[str]) -> list[str]:
    # complete Simulation definition in stages.md
    # this is just to allow the Simulation object to be instantiated...
    # as a smoke test to make sure our examples still work
    if not any(x.lstrip().startswith("sample") for x in lines):
        lines.insert(-1, "    sample={'labels': []},")
    if not any(x.lstrip().startswith("truth_space") for x in lines):
        lines.insert(-1, "    truth_space={'shape': (1,1,1), 'scale': (1,1,1)},")
    return lines
