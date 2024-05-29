import tensorstore as ts
from matplotlib import pyplot as plt

from microsim.cosem import CosemDataset

ds = CosemDataset.fetch("jrc_jurkat-1")
view = [v for v in ds.views if v.description][-1]


dataset = ts.open(view.images[0].ts_spec()).result()

plt.imshow(dataset[ts.d["x", "channel"][100, 0]].read().result())
plt.show()
