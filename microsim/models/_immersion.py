from pydantic import BaseModel


class ImmersionMedium(BaseModel):
    refractive_index: float  # could be wavelength dependent
