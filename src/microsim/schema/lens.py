import numpy as np
from pydantic import BaseModel, Field, computed_field


class ObjectiveLens(BaseModel):
    numerical_aperture: float = Field(1.4, alias="na")
    coverslip_ri: float = 1.515  # coverslip RI experimental value (ng)
    coverslip_ri_spec: float = 1.515  # coverslip RI design value (ng0)
    immersion_medium_ri: float = 1.515  # immersion medium RI experimental value (ni)
    immersion_medium_ri_spec: float = 1.515  # immersion medium RI design value (ni0)
    specimen_ri: float = 1.47  # specimen refractive index (ns)
    working_distance: float = 150.0  # um, working distance, design value (ti0)
    coverslip_thickness: float = 170.0  # um, coverslip thickness (tg)
    coverslip_thickness_spec: float = 170.0  # um, coverslip thickness design (tg0)

    @property
    @computed_field
    def half_angle(self) -> float:
        return np.arcsin(self.numerical_aperture / self.immersion_medium_ri)  # type: ignore

    @property
    @computed_field(repr=False)
    def ni(self) -> float:
        return self.immersion_medium_ri

    @property
    @computed_field(repr=False)
    def ns(self) -> float:
        return self.specimen_ri

    @property
    @computed_field(repr=False)
    def ng(self) -> float:
        return self.coverslip_ri

    @property
    @computed_field(repr=False)
    def tg(self) -> float:
        return self.coverslip_thickness * 1e-6

    @property
    @computed_field(repr=False)
    def tg0(self) -> float:
        return self.coverslip_thickness_spec * 1e-6

    @property
    @computed_field(repr=False)
    def ti0(self) -> float:
        return self.working_distance * 1e-6

    @property
    @computed_field(repr=False)
    def ng0(self) -> float:
        return self.coverslip_ri_spec

    @property
    @computed_field(repr=False)
    def ni0(self) -> float:
        return self.immersion_medium_ri_spec
