"""Dimension definitions for microsim.

1. Truth space has dims ZYX (grid on which the truth is defined)
    - here intensity is zero
2. Ground truth adds a fluorophore axis (one for each label) -> FZYX
    - here intensity values correspond to count of the fluorophore.
3. Optical image: Fluorophore axis -> Wavelength axis (emission spectrum) -> WFZYX
    - open question: do we leave F in there? (probably yes)
    - here intensity values correspond to photon count at each wavelength.
3b. Filtered Optical image applies emission filter: W axis -> C axis -> CZYX
    - each channel will bin all W (for every fluorophore) into a single value.
        - Key observation here EACH channel uses ALL Wavelengths (& fluors)
    - here intensity values correspond to photon count at each wavelength.
4. Digital image: simply applies noise and pixelation: converts photons to gray values.
    - here intensity values correspond to gray values (0-255) in the image.
"""

from enum import Enum
from typing import NamedTuple


class AxisInfo(NamedTuple):
    name: str
    key: str
    dimensionality: str | None = None
    category_type: type | str | None = None
    description: str | None = None

    @property
    def categorical(self) -> bool:
        return self.category_type is not None


class Axis(str, Enum):
    """Named axes used in microsim."""

    X = "x"
    Y = "y"
    Z = "z"  # Optical axis
    C = "c"  # Channel
    T = "t"  # Time
    F = "f"  # Fluorophore
    W = "w"  # Wavelength

    def __repr__(self) -> str:
        return f"<Axis.{self.name}>"

    def __str__(self) -> str:
        return self.value

    @property
    def info(self) -> AxisInfo:
        return AXIS_INFO[self.value]


_AXES_INFO = [
    AxisInfo(
        name="X",
        key=Axis.X.value,
        dimensionality="[length]",
        description="The x-axis, representing the horizontal axis of the image.",
    ),
    AxisInfo(
        name="Y",
        key=Axis.Y.value,
        dimensionality="[length]",
        description="The y-axis, representing the vertical axis of the image.",
    ),
    AxisInfo(
        name="Z",
        key=Axis.Z.value,
        dimensionality="[length]",
        description="The z-axis, representing the optical axis of the image.",
    ),
    AxisInfo(
        name="Time",
        key=Axis.T.value,
        dimensionality="[time]",
        description="The time axis.",
    ),
    AxisInfo(
        name="Channel",
        key=Axis.C.value,
        category_type="OpticalConfig",
        description=(
            "Channel axis in the final image. "
            "Represents different optical configurations used to acquire the image. "
            "(Not to be confused with 'Fluorophore' axis, which corresponds to the "
            "true species in the sample)"
        ),
    ),
    AxisInfo(
        name="Fluorophore",
        key=Axis.F.value,
        category_type="FluorophoreDistribution",
        description=(
            "A fluorophore (aka 'label') axis. "
            "Represents different fluorophore species in the sample. "
            "Not to be confused with 'Channel' axis (which corresponds to the final "
            "image). The fluorophore axis makes sense in a ground truth image."
        ),
    ),
    AxisInfo(
        name="Wavelength",
        key=Axis.W.value,
        dimensionality="[length]",
        description=(
            "The wavelength axis. Represents the wavelength of the light."
            "May be used to represent an emission spectrum in an image."
        ),
    ),
]

AXIS_INFO = {d.key: d for d in _AXES_INFO}
