from pydantic import BaseModel

from microsim.schema.spectrum import Spectrum


class Fluorophore(BaseModel):
    name: str
    excitation_spectrum: Spectrum
    emission_spectrum: Spectrum
    bleaching_half_life_s: float | None = None
    lifetime_ns: float | None = None
