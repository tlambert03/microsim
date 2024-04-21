from ._base_model import SimBaseModel


class Spectrum(SimBaseModel):
    wavelength: list[float]  # nm
    intensity: list[float]  # normalized to 1
    scalar: float = 1  # scalar to multiply intensity by, such as EC or QY

    @property
    def peak_wavelength(self) -> float:
        """Wavelength corresponding to maximum intensity."""
        return self.wavelength[self.intensity.index(max(self.intensity))]
