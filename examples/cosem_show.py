from microsim.cosem._dataset import CosemDataset

hela3 = CosemDataset.fetch("jrc_hela-3")
hela3.image(name="mito-mem_seg").download("~/Desktop")
# hela3.show(
#     ["mito-mem_seg", "er_seg", "fibsem-uint16"], level=3, transpose=("y", "x", "z")
# )
