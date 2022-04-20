from __future__ import annotations

from pydantic import BaseModel, Field

from ._renderable import Renderable


class Illumination(BaseModel, Renderable):
    peak_irradiance: float = Field(10, description="irradiance in W/cm2")
