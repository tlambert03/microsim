from microsim.cosem._dataset import CosemDataset

hela3 = CosemDataset.fetch("jrc_hela-3")
hela3.show(["mito-mem_seg", "er_seg", "ld_seg"], level=2, transpose=("y", "x", "z"))

