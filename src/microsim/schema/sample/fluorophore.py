from microsim.schema._base_model import SimBaseModel
from microsim.schema.spectrum import Spectrum


class Fluorophore(SimBaseModel):
    name: str
    excitation_spectrum: Spectrum
    emission_spectrum: Spectrum
    bleaching_half_life_s: float | None = None
    lifetime_ns: float | None = None
