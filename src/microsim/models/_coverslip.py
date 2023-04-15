from pydantic import BaseModel, Field


class Coverslip(BaseModel):
    ri: float = Field(1.515, description="refractive index of coverslip material")
    thickness: float = Field(170, description="thickness of coverslip in microns")
