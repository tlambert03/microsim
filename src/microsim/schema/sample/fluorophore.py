from math import log
from typing import Any, cast

import xarray as xr
from pydantic import model_validator
from scipy.constants import Avogadro

from microsim.schema._base_model import SimBaseModel
from microsim.schema.spectrum import Spectrum


class Fluorophore(SimBaseModel):
    name: str
    excitation_spectrum: Spectrum
    emission_spectrum: Spectrum
    bleaching_half_life_s: float | None = None
    extinction_coefficient: float | None = None  # M^-1 cm^-1
    quantum_yield: float | None = None
    lifetime_ns: float | None = None

    def __hash__(self) -> int:
        return id(self)

    @property
    def absorption_cross_section(self) -> xr.DataArray:
        """Return the absorption cross section in cm^2."""
        if self.extinction_coefficient is None:
            raise ValueError("Extinction coefficient is not set.")
        ec = self.excitation_spectrum.as_xarray()  # 1/cm/M
        # normalize to peak of 1
        ec = ec / ec.max()
        # multiply by extinction coefficient
        ec = ec * self.extinction_coefficient
        out = log(10) * 1e3 * ec / Avogadro  # cm^2
        out.attrs["units"] = "cm^2"
        out.attrs["long_name"] = "Absorption cross-section"
        out.name = "cross_section"
        return cast("xr.DataArray", out)

    @classmethod
    def from_fpbase(cls, name: str) -> "Fluorophore":
        from microsim.fpbase import get_fluorophore

        fpbase_fluor = get_fluorophore(name)

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

    def plot(self, show: bool = True) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(
            self.excitation_spectrum.wavelength,
            self.excitation_spectrum.intensity,
        )
        ax.plot(
            self.emission_spectrum.wavelength,
            self.emission_spectrum.intensity,
        )
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        ax.set_title(self.name)
        ax.legend(["Excitation", "Emission"])
        if show:
            plt.show()

    def all_spectra(self) -> "xr.DataArray":
        """Return a DataArray with both excitation and emission spectra."""
        da: xr.DataArray = xr.concat(
            [self.excitation_spectrum.as_xarray(), self.emission_spectrum.as_xarray()],
            dim="spectra",
        )
        da.coords.update({"spectra": [f"{self.name} {name}" for name in ["ex", "em"]]})
        return da

    def __str__(self) -> str:
        return self.name
