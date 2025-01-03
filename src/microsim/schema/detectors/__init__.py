from . import lib
from ._camera import CameraCCD, CameraCMOS, CameraEMCCD

Detector = CameraEMCCD | CameraCMOS | CameraCCD

__all__ = [
    "CameraCCD",
    "CameraCMOS",
    "CameraEMCCD",
    "Detector",
    "lib",
]
