import logging
from typing import TYPE_CHECKING, Any

from .backend import BackendName, DeviceName, NumpyAPI
from .detectors import PMT, CameraCCD, CameraCMOS, CameraEMCCD, HyD
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
    "PMT",
    "BackendName",
    "Bandpass",
    "CameraCCD",
    "CameraCMOS",
    "CameraEMCCD",
    "Confocal",
    "CosemLabel",
    "DeviceName",
    "DownscaledSpace",
    "ExtentScaleSpace",
    "Fluorophore",
    "FluorophoreDistribution",
    "HyD",
    "Identity",
    "LightSource",
    "Longpass",
    "MatsLines",
    "Modality",
    "NumpyAPI",
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
]


def __getattr__(name: str) -> Any:
    if name == "CosemLabel":
        from .sample import CosemLabel

        return CosemLabel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
