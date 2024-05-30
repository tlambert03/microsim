import numpy as np
import numpy.typing as npt
from microsim.cosem import CosemDataset
from microsim.util import view_nd

hela3 = CosemDataset.fetch("jrc_hela-3")
# hela3.image(name="mito-mem_seg").download(max_level=3)
# hela3.show(["mito-mem_seg"], level=1, transpose=("y", "x", "z"))

img = hela3.image(name="mito-mem_seg")
data = img.read(level=1, transpose=("y", "x", "z"))
print(.shape)
view_nd(img)


def convert(data: npt.NDArray, target: str = ".") -> None:
    from microsim.util import bin

    bdata = data.astype(bool)
    binned = bin(bdata, 2, dtype=np.uint8)

#     print("converted")
#     binned = bin(_vol, 2, dtype=np.uint8)
#     print("binned")
#     zarr.convenience.save_array(
#         f"{dataset}/{source}_b2.zarr", binned, chunks=(128, 128, 128)
#     )
#     print("saved")
#     binned = bin(binned, 2, dtype=np.uint8)
#     print("binned again")
#     zarr.convenience.save_array(
#         f"{dataset}/{source}_b4.zarr", binned, chunks=(64, 64, 64)
#     )
#     print(data, "DONE")
