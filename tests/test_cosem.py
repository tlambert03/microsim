from datatree import DataTree

from microsim.cosem import CosemDataset


def test_cosem() -> None:
    img = CosemDataset.get("jrc_hela-2").read_image("chrom_pred")
    assert isinstance(img, DataTree)
