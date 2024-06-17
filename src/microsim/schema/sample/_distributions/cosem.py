from functools import cached_property
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import numpy as np
from pydantic import BeforeValidator, computed_field, model_validator

from microsim.cosem.models import CosemDataset, CosemImage
from microsim.schema.backend import NumpyAPI

from ._base import _BaseDistribution

if TYPE_CHECKING:
    from microsim._data_array import xrDataArray
    from microsim.schema.space import Space


def _validate_dataset(v: Any) -> CosemDataset:
    if isinstance(v, CosemDataset):
        return v
    if isinstance(v, str):
        return CosemDataset.fetch(v)
    raise ValueError(f"Invalid dataset: {v!r}")


Dataset = Annotated[CosemDataset, BeforeValidator(_validate_dataset)]


class CosemLabel(_BaseDistribution):
    """Renders ground truth based on a specific layer from a COSEM dataset.

    Go to https://openorganelle.janelia.org/datasets/ to find a dataset, and then
    chose a label from the dataset to render.
    """

    type: Literal["cosem"] = "cosem"
    dataset: str
    label: str
    # position None implies crop to center
    position: tuple[float, float, float] | None = None

    def cache_path(self) -> tuple[str, ...] | None:
        if self.position:
            pos = "_".join(f"{p:.4f}" for p in self.position)
        else:
            pos = "center"
        return (self.type, self.dataset, self.label, pos)

    @computed_field(repr=False)  # type: ignore
    @cached_property
    def cosem_dataset(self) -> CosemDataset:
        return CosemDataset.fetch(self.dataset)

    @computed_field(repr=False)  # type: ignore
    @cached_property
    def cosem_image(self) -> CosemImage:
        try:
            return self.cosem_dataset.image(name=self.label)
        except ValueError:
            # if a description is provided, look for something with that description
            # we default to looking through prediction layers here, since they provide
            # and "interesting" ground truth.  but segmentation_layers could be used
            for layer in self.cosem_dataset.prediction_layers:
                if layer.description.lower() == self.label.lower():
                    return layer
            raise

    @model_validator(mode="after")
    def _verify(self) -> "CosemLabel":
        img = self.cosem_image  # will raise if not found
        if img.content_type not in {"segmentation", "prediction"}:
            raise ValueError(
                f"Cosem Image '{img.dataset_name}::{img.name}' has an unsupported "
                f"content type: {img.content_type!r}\n"
            )
        return self

    def render(self, space: "xrDataArray", xp: NumpyAPI | None = None) -> "xrDataArray":
        xp = xp or NumpyAPI()

        # FIXME: ugly
        truth_space = cast("Space", space.attrs["space"])

        # this whole next bit is a Hack for now.
        # what we're doing is fetching the appropriate resolution level from the
        # cosem dataset based on the scale of the truth space
        # both tensorstore and xarray have much better support for indexing into
        # coordinates, but we're not using them here yet.

        # `scale` tells us how many pixels in the truth space correspond to one pixel
        # in the cosem image
        # TODO: currently hardcoded to 0.004 um/px, but this should be taken from
        # cosem_image.grid_scale
        # TODO: interpolate if the scale is not an integer multiple of 0.004 um/px
        # FIXME: most cosem datasets have a grid scale more like:
        # z=3.24 y=4.0 x=4.0
        # TODO: use pint for units. truth_space scale is implicitly um.
        # cosem grid_scale is in nm
        scale = [round(x / 0.004, 4) for x in truth_space.scale]
        if any(x % 1.0 for x in scale):
            raise ValueError("Only 0.004 um/px multiples are currently supported")

        # scale now represents how many times larger than 4nm/px the truth space is
        # we need to get the appropriate resolution level from the cosem image
        # where lvl 0 is 4nm/px, lvl 1 is 8nm/px, etc...
        cosem_level = int(np.log2(max(scale)))
        data = self.cosem_image.read(level=cosem_level, bin_mode="auto")

        if (trans := self.position) is None:
            trans = tuple(-s // 2 for s in data.shape)
        data = data.translate_to[trans]

        dmin = data.domain.inclusive_min
        dmax = data.domain.inclusive_max
        half_size = tuple(s // 2 for s in space.shape)
        slc = tuple(
            slice(max(-s, dmi), min(s, dma))
            for dmi, dma, s in zip(dmin, dmax, half_size, strict=True)
        )

        extracted = xp.asarray(data[slc].read().result()).astype(space.dtype)
        # pad with zeros (centering) to match the space shape
        if not extracted.shape == space.shape:
            pad = tuple(
                (0, s - e) for s, e in zip(space.shape, extracted.shape, strict=False)
            )
            extracted = xp.pad(extracted, pad, mode="constant")
        return space + extracted
