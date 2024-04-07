from pathlib import Path

import pytest
import yaml

from microsim.schema import Simulation


@pytest.fixture
def confocal_yaml() -> Path:
    return Path(__file__).parent / "data" / "confocal.yaml"


def test_schema(confocal_yaml: Path):
    data = yaml.safe_load(confocal_yaml.read_text())
    sim = Simulation(**data)
    output = sim.run()
    assert output.shape == sim.output_space.shape
