import logging
from typing import TYPE_CHECKING, Any

from .aberration import ZernikeAberration
from .detectors import CameraCCD, CameraCMOS, CameraEMCCD
from .lens import ObjectiveLens
from .modality import Confocal, Identity, Modality, Widefield
from .optical_config import (
    Bandpass,
    LightSource,
    Longpass,
    OpticalConfig,
    Shortpass,
    SpectrumFilter,
)
from .sample import Fluorophore, FluorophoreDistribution, MatsLines, Sample
from .settings import Settings
from .simulation import Simulation
from .space import DownscaledSpace, ExtentScaleSpace, ShapeExtentSpace, ShapeScaleSpace
from .spectrum import Spectrum

if TYPE_CHECKING:
    from .sample._distributions.cosem import CosemLabel

logging.getLogger().setLevel(logging.INFO)
__all__ = [
    "Bandpass",
    "CameraCCD",
    "CameraCMOS",
    "CameraEMCCD",
    "Confocal",
    "CosemLabel",
    "DownscaledSpace",
    "ExtentScaleSpace",
    "Fluorophore",
    "FluorophoreDistribution",
    "Identity",
    "LightSource",
    "Longpass",
    "MatsLines",
    "Modality",
    "ObjectiveLens",
    "OpticalConfig",
    "Sample",
    "Settings",
    "ShapeExtentSpace",
    "ShapeScaleSpace",
    "Shortpass",
    "Simulation",
    "Spectrum",
    "SpectrumFilter",
    "Widefield",
    "ZernikeAberration",
]


def __getattr__(name: str) -> Any:
    if name == "CosemLabel":
        from .sample import CosemLabel

        return CosemLabel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
