import numpy as np
from datatree import DataTree

from microsim.cosem import CosemDataset


def test_cosem() -> None:
    dataset = CosemDataset.fetch("jrc_hela-2")
    assert isinstance(dataset.read_image("chrom_pred"), DataTree)
    assert isinstance(dataset.thumbnail, np.ndarray)
