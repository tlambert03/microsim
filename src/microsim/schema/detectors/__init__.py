from . import lib
from ._camera import CameraCCD, CameraCMOS, CameraEMCCD
from ._point import PMT, HyD

Detector = CameraEMCCD | CameraCMOS | CameraCCD | PMT | HyD

__all__ = [
    "PMT",
    "CameraCCD",
    "CameraCMOS",
    "CameraEMCCD",
    "Detector",
    "HyD",
    "lib",
]
