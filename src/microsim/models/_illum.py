from __future__ import annotations

from pydantic import BaseModel, Field


class Illumination(BaseModel):
    peak_irradiance: float = Field(10, description="irradiance in W/cm2")
