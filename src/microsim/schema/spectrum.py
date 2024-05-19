from typing import ClassVar

import numpy as np
from pydantic import ConfigDict

from ._base_model import SimBaseModel


class Spectrum(SimBaseModel):
    model_config: ClassVar[ConfigDict] = {
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }
    wavelength: np.ndarray  # nm
    intensity: np.ndarray  # normalized to 1
    scalar: float = 1  # scalar to multiply intensity by, such as EC or QY

    @property
    def peak_wavelength(self) -> float:
        """Wavelength corresponding to maximum intensity."""
        return self.wavelength[self.intensity.index(max(self.intensity))]
