from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from pydantic import model_validator

from ._base_model import SimBaseModel

if TYPE_CHECKING:
    from typing import Self

    import jax


# common ANSI/OSA Zernike indices, for readability when building aberrations
ANSI = {
    "vertical_tilt": 1,
    "horizontal_tilt": 2,
    "oblique_astigmatism": 3,
    "defocus": 4,
    "vertical_astigmatism": 5,
    "vertical_trefoil": 6,
    "vertical_coma": 7,
    "horizontal_coma": 8,
    "oblique_trefoil": 9,
    "primary_spherical": 12,
}


class ZernikeAberration(SimBaseModel):
    """Pupil-phase aberration expressed in the Zernike basis.

    `coefficients` are wavefront optical path differences (in microns) on the
    ANSI/OSA-indexed, RMS-normalized Zernike basis, one per entry in
    `ansi_indices`.  Because they are an OPD they are wavelength-independent;
    the per-wavelength phase is `2π·OPD/λ`.

    These coefficients are the natural variable for gradient-based phase
    retrieval and PSF engineering.
    """

    ansi_indices: tuple[int, ...] = ()
    coefficients: tuple[float, ...] = ()

    @model_validator(mode="after")
    def _check_lengths(self) -> Self:
        if len(self.ansi_indices) != len(self.coefficients):
            raise ValueError(
                "ansi_indices and coefficients must have the same length, got "
                f"{len(self.ansi_indices)} and {len(self.coefficients)}"
            )
        return self

    def as_array(self) -> jax.Array:
        """Coefficients as a JAX array (the differentiable parameter)."""
        return jnp.asarray(self.coefficients, dtype=float)

    def __bool__(self) -> bool:
        return bool(self.ansi_indices)
