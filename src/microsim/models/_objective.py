from pydantic import BaseModel, Field, model_validator


class Objective(BaseModel):
    na: float = Field(1.42, description="numerical aperture.")
    immersion_ri_design: float = Field(
        1.515, description="design (expected) refractive index of immersion medium"
    )
    working_distance: float = Field(
        150,
        description="working distance (immersion medium thickness) design, in microns.",
    )
    magnification: float = Field(1, description="magnification of objective lens.")
    cs_ri_design: float = Field(
        1.515, description="design (expected) refractive index of coverslip"
    )
    cs_thickness_design: float = Field(
        170, description="design (expected) thickness of coverslip in microns"
    )

    @model_validator(mode="before")
    def _vroot(cls, values: dict):
        na = values.get("na", 0)
        ri = values.get("immersion_ri_design", 1000)
        if na > ri:
            raise ValueError(
                f"NA ({na}) cannot be greater than the immersion medium RI "
                f"design value ({ri})"
            )
        return values
