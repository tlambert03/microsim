from __future__ import annotations

from collections import defaultdict
from enum import IntEnum
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from microsim.util import http_get

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence


class SWCType(IntEnum):
    """Constants for the SWC node types."""

    UNDEFINED = 0
    SOMA = 1
    AXON = 2
    BASAL_DENDRITE = 3
    APICAL_DENDRITE = 4
    CUSTOM = 5
    UNSPECIFIED_NEURITE = 6
    GLIA_PROCESSES = 7


class Compartment(NamedTuple):
    """A compartment in an SWC file."""

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
    """Object representing an SWC file.

    Provides methods to parse SWC files and render them as binary masks.
    """

    @classmethod
    def from_path(cls, path: str | Path) -> SWC:
        if str(path).startswith(("http://", "https://")):
            response = http_get(str(path))
            content = response.decode()
        else:
            content = Path(path).expanduser().read_text()
        return cls.from_string(content)

    @classmethod
    def from_string(cls, content: str | bytes) -> SWC:
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

    def binary_mask(
        self,
        voxel_size: float = 1,
        scale_factor: float = 3,
        *,
        include_types: Iterable[int] = (
            SWCType.BASAL_DENDRITE,
            SWCType.APICAL_DENDRITE,
            SWCType.AXON,
        ),
    ) -> np.ndarray:
        """Render a binary mask of the neuron reconstruction."""
        from microsim._draw import draw_line_3d, draw_sphere

        grid = self._empty_grid(voxel_size)
        origin = np.min(self.coords, axis=0)

        dend_scale: float = 1
        max_r = float(np.sum(grid.shape))

        for par, child in self.iter_pairs(*include_types):
            r = int(max(1, 0.5 * scale_factor * dend_scale * (par.r + child.r)))
            pz, py, px = par.shifted_coord(origin, voxel_size)
            cz, cy, cx = child.shifted_coord(origin, voxel_size)
            draw_line_3d(px, py, pz, cx, cy, cz, grid, max_r=max_r, width=r)

        soma_scale: float = 1
        for comp in self._node_types[SWCType.SOMA]:
            z, y, x = comp.shifted_coord(origin, voxel_size)
            r = int(0.5 * scale_factor * soma_scale * comp.r)
            draw_sphere(grid, x, y, z, r)

        return grid.astype(np.uint8)
