from typing import Any

from pydantic import model_validator

from microsim._field_types import ExtCoeff, Nanoseconds, Seconds
from microsim.schema._base_model import SimBaseModel
from microsim.schema.spectrum import Spectrum


class Fluorophore(SimBaseModel):
    name: str
    excitation_spectrum: Spectrum
    emission_spectrum: Spectrum
    bleaching_half_life: Seconds | None = None
    extinction_coefficient: ExtCoeff | None = None
    quantum_yield: float | None = None
    lifetime: Nanoseconds | None = None

    @classmethod
    def from_fpbase(cls, name: str) -> "Fluorophore":
        from microsim.fpbase import get_fluorophore

        try:
            fpbase_fluor = get_fluorophore(name)
        except Exception as e:
            raise ValueError(f"Unable to load fluorophore {name!r} from FPbase") from e

        if (state := fpbase_fluor.default_state) is None:
            raise ValueError(f"Fluorophore {name!r} has ")

        return cls(
            name=name,
            excitation_spectrum=state.excitation_spectrum.data,  # type: ignore
            emission_spectrum=state.emission_spectrum.data,  # type: ignore
            extinction_coefficient=state.extCoeff,
            quantum_yield=state.qy,
            lifetime_ns=state.lifetime,
        )

    @model_validator(mode="before")
    def _vmodel(cls, value: Any) -> Any:
        if isinstance(value, str):
            # TODO: seems weird to have to cast back to dict...
            # but otherwise doesn't work with 'before' validator.  look into it.
            return cls.from_fpbase(value).model_dump()
        return value
