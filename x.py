# ds =CosemDataset.fetch("jrc_jurkat-1")
# view = ds.view("Default view")
# print(view)
import tensorstore as ts

dataset = ts.open(
    {
        "driver": "neuroglancer_precomputed",
        "kvstore": {
            "driver": "s3",
            "bucket": "janelia-cosem-datasets",
            "path": "jrc_jurkat-1/neuroglancer/em/fibsem-uint8.precomputed",
        },
        "scale_index": 4,
    }
).result()
print(dataset[ts.d["z"][100]])
