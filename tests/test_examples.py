import json
import runpy
from pathlib import Path

import pytest

from microsim import util

EXAMPLE_DIR = Path(__file__).parent.parent / "examples/"
skip = {
    "illum_widget",
    "fftconv_bench",
    "sim3d",  # very slow, and not really used yet
}
examples = [
    f
    for f in EXAMPLE_DIR.glob("*")
    if f.stem not in skip and f.suffix in [".py", ".ipynb"]
]


@pytest.mark.usefixtures("mpl_show_patch")
@pytest.mark.parametrize("fpath", examples, ids=lambda x: x.name)
def test_examples(fpath: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that all of our examples are still working without warnings."""
    monkeypatch.setattr(util, "ndview", lambda *args, **kwargs: None)
    if fpath.suffix == ".ipynb":
        nb = json.loads(fpath.read_text(encoding="utf-8"))

        lines = []
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                lines.extend(cell["source"])

        fpath = tmp_path / fpath.name
        fpath.write_text("\n".join(lines))
    runpy.run_path(str(fpath), run_name="__main__")
