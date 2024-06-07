from collections import defaultdict
from collections.abc import Sequence
from enum import IntEnum
from functools import cache, cached_property
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import numpy as np
import requests
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterable

ALLEN_ROOT = "http://api.brain-map.org"
ALLEN_V2_API = f"{ALLEN_ROOT}/api/v2/data"
ALLEN_V2_QUERY = ALLEN_V2_API + "/query.json"
SWC_FILE_TYPE = "3DNeuronReconstruction"


class WellKnownFileType(BaseModel):
    id: int
    name: str


class WellKnownFile(BaseModel):
    attachable_id: int | None
    attachable_type: str | None
    download_link: str | None
    id: int | None
    path: str | None
    well_known_file_type_id: int | None
    well_known_file_type: WellKnownFileType | None


class SWCType(IntEnum):
    UNDEFINED = 0
    SOMA = 1
    AXON = 2
    BASAL_DENDRITE = 3
    APICAL_DENDRITE = 4
    CUSTOM = 5
    UNSPECIFIED_NEURITE = 6
    GLIA_PROCESSES = 7


class Compartment(NamedTuple):
    id: int
    t: int  # type
    x: float
    y: float
    z: float
    r: float  # radius
    c: int  # parent id or "connectivity"


class SWC:
    def __init__(self, compartments: Sequence[Compartment] = ()):
        self.compartments = compartments

        self.children: defaultdict[int, list[Compartment]] = defaultdict(list)
        self.node_types: defaultdict[int, list[Compartment]] = defaultdict(list)
        self._map: dict[int, Compartment] = {}
        for comp in compartments:
            self._map[comp.id] = comp
            self.children[comp.c].append(comp)
            self.node_types[comp.t].append(comp)

    def iter_pairs(self, starting_type: int):
        seen = set()
        for comp_id, children in self.children.items():
            if comp_id == -1:
                continue
            comp = self._map[comp_id]
            if comp.t == starting_type:
                for child in children:
                    if (comp, child) not in seen:
                        yield comp, child
                        seen.add((comp, child))

    @classmethod
    def from_string(cls, content: str | bytes) -> "SWC":
        if isinstance(content, bytes):
            content = content.decode()

        compartments = []
        for num, line in enumerate(content.splitlines()):
            if line.startswith("#"):
                continue
            try:
                a, b, c, d, e, f, g = line.split()
                comp = Compartment(
                    int(a), int(b), float(c), float(d), float(e), float(f), int(g)
                )
            except ValueError as e:
                raise ValueError(f"Invalid SWC line {num}: {line}") from e
            compartments.append(comp)
        return cls(compartments)

    @classmethod
    def from_url(cls, url: str) -> "SWC":
        response = requests.get(url)
        response.raise_for_status()
        return cls.from_string(response.text)

    @cached_property
    def coords(self) -> np.ndarray:
        """Return the coordinates of the compartments as (M, 3) array."""
        return np.array([(c.z, c.y, c.x) for c in self.compartments]).astype("float32")

    def empty_grid(self, resolution: float = 1.0) -> np.ndarray:
        extent = np.ptp(self.coords, axis=0)
        size = np.ceil(extent / resolution).astype(int)
        return np.zeros(size, dtype=np.uint8)

    def build_mask(
        self,
        resolution: float = 1.0,
        dend_scale: float = 2,
        global_scale_factor: float = 10,
    ):
        from microsim.schema.sample.matslines._bresenham import bres_draw_segment_3d

        grid = self.empty_grid(resolution)
        for par, child in self.iter_pairs(3):
            r = int(max(1, 0.5 * global_scale_factor * dend_scale * (par.r + child.r)))
            bres_draw_segment_3d(
                int(par.x),
                int(par.y),
                int(par.z),
                int(child.x),
                int(child.y),
                int(child.z),
                grid,
                5000,
                width=r,
            )
        return grid


class NeuronReconstruction(BaseModel):
    id: int
    specimen_id: int
    well_known_files: list[WellKnownFile] = Field(default_factory=list)

    @property
    def swc_path(self) -> SWC:
        """The SWC file for this reconstruction."""
        for f in self.well_known_files:
            if f.well_known_file_type.name == SWC_FILE_TYPE:
                return ALLEN_ROOT + f.download_link
        raise ValueError("No SWC file found for this reconstruction.")

    def load_swc(self) -> SWC:
        """Load the SWC file for this reconstruction."""
        return SWC.from_url(self.swc_path)


class Specimen(BaseModel):
    id: int
    is_cell_specimen: bool
    specimen_id_path: str
    structure_id: int
    neuron_reconstructions: list[NeuronReconstruction] = Field(default_factory=list)

    @classmethod
    @cache
    def fetch(cls, id: int) -> "Specimen":
        """Fetch this specimen from the Allen brain map API."""
        q = [
            "model::Specimen",
            f"rma::criteria[id$eq{id}],neuron_reconstructions(well_known_files)",
            "rma::include,neuron_reconstructions(well_known_files("
            f"well_known_file_type[name$eq'{SWC_FILE_TYPE}']))",
            "rma::options[num_rows$eq'all']",
        ]
        response = requests.get(ALLEN_V2_QUERY, params={"q": ",".join(q)})
        response.raise_for_status()
        qr = QueryResponse.model_validate_json(response.content)
        if not qr.success:
            raise ValueError(qr.msg)
        return cast("Specimen", qr.msg[0])


class ApiCellTypesSpecimenDetail(BaseModel):
    specimen__id: int
    structure__name: str | None
    structure__acronym: str | None
    donor__species: Literal["Homo Sapiens", "Mus musculus"]
    nr__reconstruction_type: str | None  # probably just 'full' or 'dendrite-only'
    nr__average_contraction: float | None
    nr__average_parent_daughter_ratio: float | None
    nr__max_euclidean_distance: float | None
    nr__number_bifurcations: int | None
    nr__number_stems: int | None

    def specimen(self) -> Specimen:
        return Specimen.fetch(self.specimen__id)

    @classmethod
    @cache
    def all_reconstructions(cls) -> tuple["ApiCellTypesSpecimenDetail", ...]:
        """Fetch details for all Specimens with reconstruction info."""
        q = (
            "model::ApiCellTypesSpecimenDetail",
            "rma::criteria[nr__reconstruction_type$ne'null']",
            "rma::options[num_rows$eq'all']",
        )
        response = requests.get(ALLEN_V2_QUERY, params={"q": ",".join(q)})
        response.raise_for_status()
        qr = QueryResponse.model_validate_json(response.content)
        if not qr.success:
            raise ValueError(qr.msg)
        return tuple(qr.msg)  # type: ignore[arg-type]


class QueryResponse(BaseModel):
    success: bool
    msg: list[Specimen] | list[ApiCellTypesSpecimenDetail] | str


def get_reconstructions(
    species: Literal["Homo Sapiens", "Mus musculus"] | None,
    reconstruction_type: Literal["full", "dendrite-only"] | None = None,
) -> tuple[ApiCellTypesSpecimenDetail, ...]:
    recons: Iterable = ApiCellTypesSpecimenDetail.all_reconstructions()
    if species is not None:
        recons = (x for x in recons if x.donor__species == species)
    if reconstruction_type is not None:
        recons = (x for x in recons if x.nr__reconstruction_type == reconstruction_type)
    return tuple(recons)
