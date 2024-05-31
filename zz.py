from microsim.cosem import CosemDataset
from microsim.cosem._tstore import rebin

dset = CosemDataset.fetch("jrc_hela-3")
img = dset.image(name="mito-mem_seg")

rebin(img)
