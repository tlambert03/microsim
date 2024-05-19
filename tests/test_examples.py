import json
import runpy
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).parent.parent / "examples/"
skip = {"illum_widget", "fftconv_bench"}
examples = [
    f
    for f in EXAMPLE_DIR.glob("*")
    if f.stem not in skip and f.suffix in [".py", ".ipynb"]
]


@pytest.fixture()
def _example_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)


@pytest.mark.usefixtures("_example_monkeypatch")
@pytest.mark.parametrize("fpath", examples, ids=lambda x: x.name)
def test_examples(fpath: Path, tmp_path: Path) -> None:
    """Test that all of our examples are still working without warnings."""
    if fpath.suffix == ".ipynb":
        nb = json.loads(fpath.read_text())

        lines = []
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                lines.extend(cell["source"])

        fpath = tmp_path / fpath.name
        fpath.write_text("\n".join(lines))
    runpy.run_path(str(fpath), run_name="__main__")
