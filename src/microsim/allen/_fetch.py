from __future__ import annotations

from functools import cache, cached_property
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, Field

from microsim.util import http_get

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np

    from ._swc import SWC

ALLEN_ROOT = "http://api.brain-map.org"
ALLEN_V2_API = f"{ALLEN_ROOT}/api/v2/data"
ALLEN_V2_QUERY = ALLEN_V2_API + "/query.json"
SWC_FILE_TYPE = "3DNeuronReconstruction"


class WellKnownFileType(BaseModel):
    """Model representing a well-known file type in the Allen Brain Map API."""

    id: int
    name: str  # something like '3DNeuronReconstruction'


class WellKnownFile(BaseModel):
    """Model representing a file in the Allen Brain Map API."""

    attachable_id: int | None
    attachable_type: str | None
    download_link: str | None
    id: int | None
    path: str | None
    well_known_file_type_id: int | None
    well_known_file_type: WellKnownFileType | None


class NeuronReconstruction(BaseModel):
    """Model representing a neuron reconstruction in the Allen Brain Map API."""

    id: int
    specimen_id: int
    number_nodes: int
    number_branches: int
    number_stems: int
    number_bifurcations: int
    max_euclidean_distance: float
    neuron_reconstruction_type: str
    overall_height: float
    overall_width: float
    overall_depth: float
    scale_factor_x: float
    scale_factor_y: float
    scale_factor_z: float
    total_length: float
    total_surface: float
    total_volume: float
    well_known_files: list[WellKnownFile] = Field(default_factory=list)

    @property
    def swc_path(self) -> str:
        """The SWC file for this reconstruction."""
        for f in self.well_known_files:
            if (
                getattr(f.well_known_file_type, "name", None) == SWC_FILE_TYPE
                and f.download_link
            ):
                return ALLEN_ROOT + f.download_link
        raise ValueError(
            "No SWC file found for this reconstruction."
        )  # pragma: no cover

    @cached_property
    def swc(self) -> SWC:
        """Load the SWC file for this reconstruction."""
        from ._swc import SWC

        return SWC.from_path(self.swc_path)

    def binary_mask(self, voxel_size: float = 1, scale_factor: float = 3) -> np.ndarray:
        """Return 3D binary mask for this neuron reconstructions."""
        return self.swc.binary_mask(voxel_size=voxel_size, scale_factor=scale_factor)

    @classmethod
    @cache
    def fetch(cls, id: int) -> NeuronReconstruction:
        """Fetch NeuronReconstruction by ID from the Allen brain map API."""
        q = [
            "model::NeuronReconstruction",
            f"rma::criteria[id$eq{id}],well_known_files",
            f"rma::include,well_known_files(well_known_file_type[name$eq'{SWC_FILE_TYPE}'])",
            # get all rows
            "rma::options[num_rows$eq'all']",
        ]
        response = http_get(ALLEN_V2_QUERY, params={"q": ",".join(q)})
        qr = _QueryResponse.model_validate_json(response)
        if not qr.success:  # pragma: no cover
            raise ValueError(qr.msg)
        return cast("NeuronReconstruction", qr.msg[0])

    def specimen(self) -> Specimen:
        """Fetch the specimen that owns this neuron reconstruction."""
        return Specimen.fetch(self.specimen_id)


class Structure(BaseModel):
    """Speciment structure model from the Allen Brain Map API."""

    id: int
    name: str
    acronym: str
    structure_id_path: str


class Specimen(BaseModel):
    """Model representing a specimen in the Allen Brain Map API."""

    id: int
    name: str
    is_cell_specimen: bool
    specimen_id_path: str
    structure: Structure
    neuron_reconstructions: list[NeuronReconstruction] = Field(default_factory=list)

    @classmethod
    @cache
    def fetch(cls, id: int) -> Specimen:
        """Fetch this specimen from the Allen brain map API."""
        q = [
            # query the Specimen model
            "model::Specimen",
            # limit to the specimen with the given ID
            # and join on NeuronReconstruction and WellKnownFile
            f"rma::criteria[id$eq{id}],neuron_reconstructions(well_known_files)",
            # include structure
            # and neuron_reconstructions where the well_known_file_type is SWC
            "rma::include,structure,neuron_reconstructions(well_known_files("
            f"well_known_file_type[name$eq'{SWC_FILE_TYPE}']))",
            # get all rows
            "rma::options[num_rows$eq'all']",
        ]
        response = http_get(ALLEN_V2_QUERY, params={"q": ",".join(q)})
        qr = _QueryResponse.model_validate_json(response)
        if not qr.success:  # pragma: no cover
            raise ValueError(qr.msg)
        return cast("Specimen", qr.msg[0])

    def binary_masks(
        self, voxel_size: float = 1, scale_factor: float = 3
    ) -> list[np.ndarray]:
        """Return all binary masks for this specimen's neuron reconstructions."""
        masks = []
        for recon in self.neuron_reconstructions:
            masks.append(
                recon.binary_mask(voxel_size=voxel_size, scale_factor=scale_factor)
            )
        return masks

    @property
    def url(self) -> str:
        """Return the URL for this specimen on the Allen Brain Map."""
        return f"http://celltypes.brain-map.org/experiment/morphology/{self.id}"

    def open_webpage(self) -> None:  # pragma: no cover
        """Open the webpage for this specimen in the Allen Brain Map."""
        import webbrowser

        webbrowser.open(self.url)


class ApiCellTypesSpecimenDetail(BaseModel):
    """Model representing Specimen details from the Allen Brain Map API."""

    specimen__id: int
    structure__name: str | None
    structure__acronym: str | None
    donor__species: Literal["Homo Sapiens", "Mus musculus"]
    nr__reconstruction_type: str | None  # probably just 'full' or 'dendrite-only'
    nr__max_euclidean_distance: float | None
    nr__number_bifurcations: int | None
    nr__number_stems: int | None

    @classmethod
    @cache
    def all_reconstructions(cls) -> tuple[ApiCellTypesSpecimenDetail, ...]:
        """Fetch details for all Specimens with reconstruction info."""
        q = (
            "model::ApiCellTypesSpecimenDetail",
            "rma::criteria[nr__reconstruction_type$ne'null']",
            "rma::options[num_rows$eq'all']",
        )
        response = http_get(ALLEN_V2_QUERY, params={"q": ",".join(q)})
        qr = _QueryResponse.model_validate_json(response)
        if not qr.success:  # pragma: no cover
            raise ValueError(qr.msg)
        return tuple(qr.msg)  # type: ignore[arg-type]

    def specimen(self) -> Specimen:
        """Return associated Specimen object."""
        return Specimen.fetch(self.specimen__id)


class _QueryResponse(BaseModel):
    """Query response from the Allen Brain Map API."""

    success: bool
    msg: (
        list[NeuronReconstruction]
        | list[Specimen]
        | list[ApiCellTypesSpecimenDetail]
        | str
    )


def get_reconstructions(
    species: Literal["Homo Sapiens", "Mus musculus"] | None = None,
    reconstruction_type: Literal["full", "dendrite-only"] | None = None,
) -> tuple[ApiCellTypesSpecimenDetail, ...]:
    recons: Iterable = ApiCellTypesSpecimenDetail.all_reconstructions()
    if species is not None:
        recons = (x for x in recons if x.donor__species == species)
    if reconstruction_type is not None:
        recons = (x for x in recons if x.nr__reconstruction_type == reconstruction_type)
    return tuple(recons)
