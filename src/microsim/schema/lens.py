from typing import Any, TypedDict

import numpy as np
from pydantic import Field, model_validator

from microsim._field_types import Microns

from ._base_model import SimBaseModel


class ObjectiveKwargs(TypedDict, total=False):
    numerical_aperture: float
    coverslip_ri: float
    coverslip_ri_spec: float
    immersion_medium_ri: float
    immersion_medium_ri_spec: float
    specimen_ri: float
    working_distance: float
    coverslip_thickness: float
    coverslip_thickness_spec: float
    magnification: float


class ObjectiveLens(SimBaseModel):
    numerical_aperture: float = Field(1.4, alias="na")
    coverslip_ri: float = 1.515  # coverslip RI experimental value (ng)
    coverslip_ri_spec: float = 1.515  # coverslip RI design value (ng0)
    immersion_medium_ri: float = 1.515  # immersion medium RI experimental value (ni)
    immersion_medium_ri_spec: float = 1.515  # immersion medium RI design value (ni0)
    specimen_ri: float = 1.47  # specimen refractive index (ns)
    # um, working distance, design value (ti0)
    working_distance: Microns = 150.0  # type: ignore
    # um, coverslip thickness (tg)
    coverslip_thickness: Microns = 170.0  # type: ignore
    # um, coverslip thickness design (tg0)
    coverslip_thickness_spec: Microns = 170.0  # type: ignore

    magnification: float = Field(1, description="magnification of objective lens.")

    def cache_key(self) -> str:
        """Persistent identifier for the model."""
        out = ""
        for _, val in sorted(self.model_dump(mode="python").items()):
            val = getattr(val, "magnitude", val)
            out += f"_{str(val).replace('.', '-')}"
        return out

    def __hash__(self) -> int:
        return hash(
            (
                self.numerical_aperture,
                self.coverslip_ri,
                self.coverslip_ri_spec,
                self.immersion_medium_ri,
                self.immersion_medium_ri_spec,
                self.specimen_ri,
                self.working_distance,
                self.coverslip_thickness,
                self.coverslip_thickness_spec,
                self.magnification,
            )
        )

    @model_validator(mode="before")
    def _vroot(cls, values: Any) -> Any:
        if isinstance(values, dict):
            na = values.get("numerical_aperture", 1.4)
            ri = values.get("immersion_medium_ri_spec", 1.515)
            if na > ri:
                raise ValueError(
                    f"NA ({na}) cannot be greater than the immersion medium RI "
                    f"design value ({ri})"
                )
        return values

    @property
    def half_angle(self) -> float:
        return np.arcsin(self.numerical_aperture / self.immersion_medium_ri)  # type: ignore

    @property
    def ni(self) -> float:
        return self.immersion_medium_ri

    @property
    def ns(self) -> float:
        return self.specimen_ri

    @property
    def ng(self) -> float:
        return self.coverslip_ri

    @property
    def tg(self) -> float:
        return float(self.coverslip_thickness.to("meters").magnitude)

    @property
    def tg0(self) -> float:
        return float(self.coverslip_thickness_spec.to("meters").magnitude)

    @property
    def ti0(self) -> float:
        return float(self.working_distance.to("meters").magnitude)

    @property
    def ng0(self) -> float:
        return self.coverslip_ri_spec

    @property
    def ni0(self) -> float:
        return self.immersion_medium_ri_spec
