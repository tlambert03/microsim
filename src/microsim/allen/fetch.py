from collections import defaultdict
from collections.abc import Iterator, Sequence
from enum import IntEnum
from functools import cache, cached_property
from pathlib import Path
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

    def coord(self) -> np.ndarray:
        return np.array([self.z, self.y, self.x])

    def shifted_coord(
        self, origin: np.ndarray, resolution: float = 1
    ) -> tuple[int, int, int]:
        shifted_coord = (self.coord() - origin) / resolution
        return tuple(c for c in shifted_coord.astype(int))


class SWC:
    @classmethod
    def from_path(cls, path: str | Path) -> "SWC":
        if str(path).startswith(("http://", "https://")):
            response = requests.get(str(path))
            response.raise_for_status()
            content = response.text
        else:
            content = Path(path).expanduser().read_text()
        return cls.from_string(content)

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

    def __init__(self, compartments: Sequence[Compartment] = ()):
        self.compartments = compartments

        self._id_map: dict[int, Compartment] = {}
        self._children_of: defaultdict[int, list[Compartment]] = defaultdict(list)
        self._node_types: defaultdict[int, list[Compartment]] = defaultdict(list)
        for comp in compartments:
            self._id_map[comp.id] = comp
            self._children_of[comp.c].append(comp)
            self._node_types[comp.t].append(comp)

    def iter_pairs(self, *types: int) -> Iterator[tuple[Compartment, Compartment]]:
        seen = set()
        for comp_id, children in self._children_of.items():
            if comp_id == -1:
                continue
            comp = self._id_map[comp_id]
            if not types or comp.t in types:
                for child in children:
                    if (comp, child) not in seen:
                        yield comp, child
                        seen.add((comp, child))

    @cached_property
    def coords(self) -> np.ndarray:
        """Return the coordinates of the compartments as (M, 3) array."""
        return np.array([(c.z, c.y, c.x) for c in self.compartments]).astype("float32")

    def root(self) -> Compartment:
        """Return the root compartment of the SWC."""
        return self._id_map[1]

    def origin(self) -> tuple[float, float, float]:
        """Return the (Z,Y,X) coordinate of the root node."""
        root = self.root()
        return root.z, root.y, root.x

    def _empty_grid(self, voxel_size: float = 1.0) -> np.ndarray:
        """Create an empty 3D grid for the binary mask."""
        extent = np.ptp(self.coords, axis=0)
        size = np.ceil(extent / voxel_size).astype(int)
        return np.zeros(size, dtype=bool)

    def binary_mask(self, voxel_size: float = 1, scale_factor: float = 3) -> np.ndarray:
        from microsim._draw import draw_line_3d, draw_sphere

        grid = self._empty_grid(voxel_size)
        origin = np.min(self.coords, axis=0)

        dend_scale: float = 1
        max_r = float(np.sum(grid.shape))
        for par, child in self.iter_pairs(
            SWCType.BASAL_DENDRITE, SWCType.APICAL_DENDRITE
        ):
            r = int(max(1, 0.5 * scale_factor * dend_scale * (par.r + child.r)))
            pz, py, px = par.shifted_coord(origin, voxel_size)
            cz, cy, cx = child.shifted_coord(origin, voxel_size)
            draw_line_3d(px, py, pz, cx, cy, cz, grid, max_r=max_r, width=r)

        soma_scale: float = 1.2
        for comp in self._node_types[SWCType.SOMA]:
            z, y, x = comp.shifted_coord(origin, voxel_size)
            r = int(0.5 * scale_factor * soma_scale * comp.r)
            draw_sphere(grid, x, y, z, r)

        return grid.astype(np.uint8)


class NeuronReconstruction(BaseModel):
    id: int
    specimen_id: int
    number_nodes: int
    number_branches: int
    neuron_reconstruction_type: str
    overall_height: float
    overall_width: float
    overall_depth: float
    scale_factor_x: float
    scale_factor_y: float
    scale_factor_z: float
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
        raise ValueError("No SWC file found for this reconstruction.")

    def load_swc(self) -> SWC:
        """Load the SWC file for this reconstruction."""
        return SWC.from_path(self.swc_path)


class Structure(BaseModel):
    id: int
    name: str
    acronym: str
    structure_id_path: str


class Specimen(BaseModel):
    id: int
    name: str
    is_cell_specimen: bool
    specimen_id_path: str
    structure: Structure
    neuron_reconstructions: list[NeuronReconstruction] = Field(default_factory=list)

    @classmethod
    @cache
    def fetch(cls, id: int) -> "Specimen":
        """Fetch this specimen from the Allen brain map API."""
        q = [
            "model::Specimen",
            f"rma::criteria[id$eq{id}],neuron_reconstructions(well_known_files)",
            "rma::include,structure,neuron_reconstructions(well_known_files("
            f"well_known_file_type[name$eq'{SWC_FILE_TYPE}']))",
            "rma::options[num_rows$eq'all']",
        ]
        response = requests.get(ALLEN_V2_QUERY, params={"q": ",".join(q)})
        response.raise_for_status()
        qr = QueryResponse.model_validate_json(response.content)
        if not qr.success:
            raise ValueError(qr.msg)
        return cast("Specimen", qr.msg[0])

    def binary_mask(self, voxel_size: float = 1, scale_factor: float = 1) -> np.ndarray:
        """Return 3D binary mask for this specimen's neuron reconstructions."""
        for recon in self.neuron_reconstructions:
            return recon.load_swc().binary_mask(
                voxel_size=voxel_size, scale_factor=scale_factor
            )
        raise ValueError("No neuron reconstructions found for this specimen.")


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
