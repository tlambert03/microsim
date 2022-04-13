from pydantic import BaseModel, Field, root_validator


class Objective(BaseModel):
    na: float = Field(..., description="numerical aperture.")
    immersion_ri: float = Field(
        ..., description="design (expected) refractive index of immersion medium"
    )
    working_distance: float = Field(
        ...,
        description="working distance (immersion medium thickness) design, in microns.",
    )
    magnification: float = Field(1, description="magnification of objective lens.")
    cs_ri_design: float = Field(
        1.515, description="design (expected) refractive index of coverslip"
    )
    cs_thickness_design: float = Field(
        170, description="design (expected) thickness of coverslip in microns"
    )

    @root_validator
    def _vroot(cls, values: dict):
        na = values.get("na", 0)
        ri = values.get("immersion_ri", 1000)
        if na > ri:
            raise ValueError(
                f"NA ({na}) cannot be greater than immersion medium RI ({ri})"
            )

        return values

    @classmethod
    def default(cls):
        return cls(na=1.4, immersion_ri=1.515, working_distance=150, magnification=1)
