from pydantic import BaseModel


class ImmersionMedium(BaseModel):
    refractive_index: float  # could be wavelength dependent


class OilImmersion(ImmersionMedium):
    refractive_index: float = 1.515
